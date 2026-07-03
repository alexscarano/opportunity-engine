# -*- coding: utf-8 -*-
"""
This module handles all interactions with Google APIs, including authentication,
file and folder management in Google Drive, and interactions with Google Slides
and Google Sheets.
"""

import os
import gspread
import google.auth.transport.requests as requests
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
import google.generativeai as genai

# Define the scope for Google services
SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/presentations",
    "https://www.googleapis.com/auth/spreadsheets",
]
TOKEN_FILE = "token.json"
CLIENT_SECRETS_FILE = "client_secrets.json"


def authenticate_google_services():
    """
    Handles local user authentication for Google APIs.
    - Checks for an existing token.json.
    - If not found or invalid, it initiates the OAuth2 flow using client_secrets.json.
    - Returns authenticated services for Drive, Slides, and Sheets.
    """
    creds = None
    try:
        if os.path.exists(TOKEN_FILE):
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(CLIENT_SECRETS_FILE):
                    raise FileNotFoundError(
                        f"ERROR: The '{CLIENT_SECRETS_FILE}' was not found. "
                        "Please download it from your Google Cloud project's OAuth 2.0 "
                        "Client IDs page and place it in the same directory as this script."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    CLIENT_SECRETS_FILE, SCOPES
                )
                creds = flow.run_local_server(port=0)
            with open(TOKEN_FILE, "w") as token:
                token.write(creds.to_json())

        drive_service = build("drive", "v3", credentials=creds)
        slides_service = build("slides", "v1", credentials=creds)
        sheets_service = build("sheets", "v4", credentials=creds)
        gc = gspread.authorize(creds)
        print("Google services authenticated successfully.")
        return drive_service, slides_service, sheets_service, gc
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None, None, None
    except Exception as e:
        print(f"An unexpected error occurred during authentication: {e}")
        return None, None, None, None


def authenticate_gemini(api_key=None, model_name=None):
    """Authenticates the Gemini client using an API key and selects the best available model."""
    if api_key is None:
        api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        print(
            "ERROR: No Gemini API key found. Please set the GEMINI_API_KEY environment variable."
        )
        return None

    try:
        genai.configure(api_key=api_key)

        # Detect available models supporting generateContent
        available_models = []
        try:
            for m in genai.list_models():
                if "generateContent" in m.supported_generation_methods:
                    available_models.append(m.name)
            # Filter out deprecated models
            available_models = [
                m
                for m in available_models
                if not any(
                    dep in m for dep in ["-2.0-", "-1.5-", "gemini-2.0", "gemini-1.5"]
                )
            ]
        except Exception as list_err:
            print(
                f"Warning: Could not list models: {list_err}. Defaulting to fallbacks."
            )

        selected_model = (
            model_name if model_name else "gemini-3.5-flash"
        )  # Default fallback

        if available_models:
            print(f"Available models: {available_models}")
            if model_name:
                # Validate the requested model name is available (either exact or ends with)
                matched = False
                for a_model in available_models:
                    if a_model == model_name or a_model.endswith(f"/{model_name}"):
                        selected_model = a_model
                        matched = True
                        break
                if not matched:
                    print(
                        f"Warning: Requested model '{model_name}' not found in available models. Selecting a priority fallback instead."
                    )
                    model_name = None

            if not model_name:
                priority = [
                    "gemini-3.5-flash",
                    "gemini-2.5-flash",
                    "gemini-2.5-pro",
                    "gemini-3.1-flash-lite",
                ]
                found = False
                for p_model in priority:
                    for a_model in available_models:
                        if a_model == p_model or a_model.endswith(f"/{p_model}"):
                            selected_model = a_model
                            found = True
                            break
                    if found:
                        break
                if not found:
                    selected_model = available_models[0]

        print(f"Selecting model: {selected_model}")
        gemini_client = genai.GenerativeModel(selected_model)
        print("Gemini client authenticated successfully.")
        return gemini_client
    except Exception as e:
        print(f"An unexpected error occurred during Gemini authentication: {e}")
        return None


def get_or_create_folder_id(drive_service, folder_path):
    """Finds a folder by path in Google Drive, creating it if it doesn't exist."""
    parent_id = "root"
    try:
        folders = folder_path.strip("/").split("/")
        for folder_name in folders:
            if not folder_name:
                continue
            query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and '{parent_id}' in parents and trashed=false"
            response = (
                drive_service.files()
                .list(q=query, spaces="drive", fields="files(id, name)")
                .execute()
            )
            found_folders = response.get("files", [])
            if found_folders:
                parent_id = found_folders[0].get("id")
            else:
                file_metadata = {
                    "name": folder_name,
                    "mimeType": "application/vnd.google-apps.folder",
                    "parents": [parent_id],
                }
                new_folder = (
                    drive_service.files()
                    .create(body=file_metadata, fields="id")
                    .execute()
                )
                parent_id = new_folder.get("id")
        return parent_id
    except HttpError as e:
        print(
            f"A Google Drive API error occurred while finding/creating the output folder: {e}"
        )
        return None


def download_file_from_drive(drive_service, file_id, destination):
    """Downloads a file from Google Drive."""
    try:
        print(f"   - Downloading file with ID: {file_id}")
        request = drive_service.files().get_media(fileId=file_id)
        with open(destination, "wb") as f:
            f.write(request.execute())
        print(f"   - Downloaded successfully to '{destination}'")
        return True
    except HttpError as e:
        print(f"A Google Drive API error occurred while downloading the file: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during file download: {e}")
        return False


def read_sheet_data(gc, sheet_id):
    """Reads all data from the first sheet of a Google Sheet."""
    try:
        print(f"   - Reading data from Google Sheet with ID: {sheet_id}")
        spreadsheet = gc.open_by_key(sheet_id)
        worksheet = spreadsheet.sheet1
        data = worksheet.get_all_values()
        print(f"   - Read {len(data)} rows successfully.")
        return data
    except gspread.exceptions.SpreadsheetNotFound:
        print(
            f"ERROR: The Google Sheet with ID '{sheet_id}' was not found. Please check the ID and your permissions."
        )
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading the Google Sheet: {e}")
        return None
