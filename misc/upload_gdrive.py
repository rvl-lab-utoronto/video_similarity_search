import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from apiclient.http import MediaFileUpload

SOURCE_CODE_DIR = os.path.dirname(os.path.abspath(__file__))

GOOGLE_FOLDER_MAPPING = {
    'evaluate': '1ZtVTqSOtFEetfav3fntIkJ4GnxfLWTKl'
}

SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly',
            'https://www.googleapis.com/auth/drive']

def upload_file_to_gdrive(file_path, folder_name):
    # If modifying these scopes, delete the file token.pickle.
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists(os.path.join(SOURCE_CODE_DIR, 'token.pickle')):
        with open(os.path.join(SOURCE_CODE_DIR, 'token.pickle'), 'rb') as token:
            creds = pickle.load(token)

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(os.path.join(SOURCE_CODE_DIR,'credentials.json'), SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(os.path.join(SOURCE_CODE_DIR, 'token.pickle'), 'wb') as token:
            pickle.dump(creds, token)

    service = build('drive', 'v3', credentials=creds)
    results = service.files().list(pageSize=10).execute()
    folder_id = GOOGLE_FOLDER_MAPPING['evaluate']

    file_metadata = {'name': os.path.basename(file_path),
                    'parents': [folder_id]}

    media = MediaFileUpload(file_path)
    file = service.files().create(body=file_metadata,
                                media_body=media,
                                fields='id').execute()
    print(file.get('id'))


if __name__ == '__main__':
    upload_file_to_gdrive('test.png', 'evaluate')
