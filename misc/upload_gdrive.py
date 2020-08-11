import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from apiclient.http import MediaFileUpload

SOURCE_CODE_DIR = os.path.dirname(os.path.abspath(__file__))


class GoogleDriveUploader:
    def __init__(self):
        self._GOOGLE_FOLDER_MAPPING = {
                        'evaluate': '1ZtVTqSOtFEetfav3fntIkJ4GnxfLWTKl',
                        'pose_estimation': '1r8zyL9LjHQcsaff-x7c7DTTpYUzWRBlr',
                        'maskrcnn_fg' : '1l80Nu24y4hzFQR8wOcikunVYSujSWRi0',
                        'maskrcnn_kp' : '1MU1283fBf3LV5LdxJe9k5obJEkW7AWGS'
                        }

        self._SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly',
                        'https://www.googleapis.com/auth/drive']

        self.build_service()

    def build_service(self):
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
                flow = InstalledAppFlow.from_client_secrets_file(os.path.join(SOURCE_CODE_DIR,'credentials.json'), self._SCOPES)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open(os.path.join(SOURCE_CODE_DIR, 'token.pickle'), 'wb') as token:
                pickle.dump(creds, token)

        self.service = build('drive', 'v3', credentials=creds)
        # return service


    def upload_file_to_gdrive(self, file_path, folder_name, file_name=None):
        # service = build_service()
        results = self.service.files().list(pageSize=10).execute()
        folder_id = self._GOOGLE_FOLDER_MAPPING[folder_name]
        if file_name is None:
            file_name = os.path.basename(file_path)
        file_metadata = {'name': file_name,
                        'parents': [folder_id]}

        media = MediaFileUpload(file_path)
        file = self.service.files().create(body=file_metadata,
                                    media_body=media,
                                    fields='id').execute()
        # print(file.get('id'))
        # return file.get('id')


    def create_new_folder_on_gdrive(self, folder_name, parent_folder=None):
        # service = build_service()
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }

        if parent_folder is not None:
            folder_id = self._GOOGLE_FOLDER_MAPPING[parent_folder]
            file_metadata['parents'] = [folder_id]

        file = self.service.files().create(body=file_metadata,fields='id').execute()
        self._GOOGLE_FOLDER_MAPPING[folder_name] = file.get('id')
        print('folder: {} is created in parent_folder:{}'.format(folder_name, parent_folder))
        # return file.get('id')


if __name__ == '__main__':
    # upload_file_to_gdrive('test.png', 'evaluate')
    # create_new_folder_on_gdrive('nani', parent_folder='pose_estimation')
    gdrive_service = GoogleDriveUploader()
    id = gdrive_service.create_new_folder_on_gdrive('nani', parent_folder='pose_estimation')
    print(id)
