from datetime import datetime
import json
import os

class ChatHistory:
    def __init__(self, storage_path='data/chat_history.json'):
        self.storage_path = storage_path
        self.history = self._load_history()

    def _load_history(self):
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_history(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def add_message(self, user_id, message_data):
        if str(user_id) not in self.history:
            self.history[str(user_id)] = []
        self.history[str(user_id)].append({
            **message_data,
            'timestamp': datetime.now().isoformat()
        })
        self._save_history()

    def get_recent_history(self, user_id, limit=5):
        return self.history.get(str(user_id), [])[-limit:]

    def add_rating(self, message_id, rating):
        for user_history in self.history.values():
            for message in user_history:
                if message.get('message_id') == message_id:
                    message['rating'] = rating
                    self._save_history()
                    return True
        return False

    def add_correction(self, message_id, correction):
        for user_history in self.history.values():
            for message in user_history:
                if message.get('message_id') == message_id:
                    message['correction'] = correction
                    self._save_history()
                    return True
        return False 