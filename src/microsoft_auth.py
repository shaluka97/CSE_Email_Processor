"""Microsoft OAuth authentication for OneDrive/SharePoint access.

Responsibilities
----------------
- OAuth2 authentication with Azure AD using MSAL
- Token caching and automatic refresh
- Provide authenticated session for SharePoint/OneDrive downloads

Requires Azure AD App Registration:
1. Go to https://portal.azure.com → Azure Active Directory → App registrations
2. Create new registration with redirect URI: http://localhost
3. Under "Authentication", enable "Allow public client flows"
4. Copy the Application (client) ID to MS_CLIENT_ID env var
5. Copy the Directory (tenant) ID to MS_TENANT_ID env var (or use "common" for multi-tenant)
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

import msal
import requests

from src.config import MicrosoftConfig, settings

logger = logging.getLogger(__name__)


class MicrosoftAuthenticator:
    """Handles Microsoft OAuth2 authentication using MSAL.

    Usage
    -----
    >>> auth = MicrosoftAuthenticator()
    >>> session = auth.get_authenticated_session()
    >>> response = session.get("https://graph.microsoft.com/v1.0/me")
    """

    def __init__(self, config: Optional[MicrosoftConfig] = None) -> None:
        self._config = config or settings.microsoft
        self._token_cache = msal.SerializableTokenCache()
        self._load_token_cache()
        self._app = self._build_app()
        self._session: Optional[requests.Session] = None

    def _build_app(self) -> msal.PublicClientApplication:
        """Build MSAL public client application."""
        authority = f"https://login.microsoftonline.com/{self._config.tenant_id}"

        return msal.PublicClientApplication(
            client_id=self._config.client_id,
            authority=authority,
            token_cache=self._token_cache,
        )

    def _load_token_cache(self) -> None:
        """Load token cache from disk if it exists."""
        cache_path = self._config.token_cache_path
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    self._token_cache.deserialize(f.read())
                logger.debug("Loaded Microsoft token cache from %s", cache_path)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load token cache: %s", exc)

    def _save_token_cache(self) -> None:
        """Persist token cache to disk."""
        if self._token_cache.has_state_changed:
            cache_path = self._config.token_cache_path
            with open(cache_path, "w") as f:
                f.write(self._token_cache.serialize())
            logger.debug("Saved Microsoft token cache to %s", cache_path)

    def get_access_token(self) -> Optional[str]:
        """Get a valid access token, refreshing or re-authenticating as needed.

        Returns
        -------
        Access token string, or None if authentication fails.
        """
        # Check for cached accounts
        accounts = self._app.get_accounts()

        if accounts:
            # Try to get token silently (from cache or refresh)
            result = self._app.acquire_token_silent(
                scopes=self._config.scopes,
                account=accounts[0],
            )
            if result and "access_token" in result:
                logger.debug("Acquired token silently from cache")
                self._save_token_cache()
                return result["access_token"]

        # No cached token available - need interactive login
        logger.info("No cached Microsoft token — launching interactive login...")
        result = self._app.acquire_token_interactive(
            scopes=self._config.scopes,
            prompt="select_account",
        )

        if result and "access_token" in result:
            logger.info("Microsoft authentication successful")
            self._save_token_cache()
            return result["access_token"]

        # Authentication failed
        error = result.get("error", "unknown")
        error_desc = result.get("error_description", "No description")
        logger.error("Microsoft authentication failed: %s - %s", error, error_desc)
        return None

    def get_authenticated_session(self) -> Optional[requests.Session]:
        """Get a requests.Session with Microsoft OAuth bearer token.

        Returns
        -------
        Authenticated Session, or None if authentication fails.
        """
        token = self.get_access_token()
        if not token:
            return None

        session = requests.Session()
        session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        })

        self._session = session
        return session

    def is_configured(self) -> bool:
        """Check if Microsoft OAuth is configured with required settings."""
        return bool(self._config.client_id)

    def download_sharepoint_file(
        self,
        sharing_url: str,
        timeout: int = 30,
    ) -> Optional[bytes]:
        """Download a file from a SharePoint sharing URL using Graph API.

        Parameters
        ----------
        sharing_url:
            The SharePoint/OneDrive sharing URL.
        timeout:
            Request timeout in seconds.

        Returns
        -------
        File bytes, or None on failure.
        """
        session = self.get_authenticated_session()
        if not session:
            logger.error("Cannot download SharePoint file: authentication failed")
            return None

        # Convert sharing URL to Graph API driveItem endpoint
        # The sharing URL needs to be base64 encoded for the Graph API
        import base64

        # Encode the sharing URL for Graph API
        encoded_url = base64.urlsafe_b64encode(sharing_url.encode()).decode()
        # Remove padding and add 'u!' prefix
        encoded_url = "u!" + encoded_url.rstrip("=")

        # Use Graph API to get the driveItem
        graph_url = f"https://graph.microsoft.com/v1.0/shares/{encoded_url}/driveItem"

        try:
            # First, get the driveItem metadata to find the download URL
            response = session.get(graph_url, timeout=timeout)
            response.raise_for_status()

            item_data = response.json()
            download_url = item_data.get("@microsoft.graph.downloadUrl")

            if not download_url:
                logger.error("No download URL in driveItem response")
                return None

            # Download the actual file content
            logger.debug("Downloading from Graph API: %s", download_url[:100])
            file_response = session.get(download_url, timeout=timeout)
            file_response.raise_for_status()

            return file_response.content

        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response else "?"
            logger.error(
                "Graph API error (%s) downloading SharePoint file: %s",
                status,
                sharing_url,
            )
            if exc.response is not None:
                logger.debug("Response: %s", exc.response.text[:500])
            return None
        except requests.exceptions.RequestException as exc:
            logger.error("Request failed for SharePoint file %s: %s", sharing_url, exc)
            return None


# Module-level singleton for convenience
_authenticator: Optional[MicrosoftAuthenticator] = None


def get_microsoft_authenticator() -> MicrosoftAuthenticator:
    """Get or create the Microsoft authenticator singleton."""
    global _authenticator
    if _authenticator is None:
        _authenticator = MicrosoftAuthenticator()
    return _authenticator
