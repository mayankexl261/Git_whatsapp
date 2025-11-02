import fakeredis
import json
 
# r = redis.Redis(host="0.0.0.0", port=6379, db=0, decode_responses=True)
r = fakeredis.FakeStrictRedis()
def get_conversation(session_id):
    data = r.get(session_id)
    if data:
        return json.loads(data)
    return []
 
def save_conversation(session_id, messages):
    r.set(session_id, json.dumps(messages))