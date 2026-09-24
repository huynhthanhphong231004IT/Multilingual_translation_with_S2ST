import time
from collections import deque

def identifier(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ValueError(f'{name} must be a non-empty string (max 512 characters)')
    return value

def room_id_from_metadata(metadata):
    if not isinstance(metadata, dict):
        raise ValueError('Metadata must be a JSON object')
    return identifier(metadata.get('internalMeetingId'), 'internalMeetingId')

class Room:
    def __init__(self, internal_meeting_id):
        self.internal_meeting_id = internal_meeting_id
        self.subscribers = set()
        self.streams = set()
        self.partials = {}
        self.history = deque(maxlen=100)
        self.updated = time.monotonic()
        self.sequence = 0

    def snapshot(self):
        return {'type': 'snapshot', 'internalMeetingId': self.internal_meeting_id,
                'session': {'internalMeetingId': self.internal_meeting_id,
                            'status': 'connected' if self.streams else 'disconnected',
                            'activeStreams': len(self.streams)},
                'partials': list(self.partials.values()), 'history': list(self.history)}

    def publish(self, event):
        self.updated = time.monotonic()
        self.sequence += 1
        event = {**event, 'internalMeetingId': self.internal_meeting_id, 'sequence': self.sequence}
        stream = event.get('sessionId')
        if event['type'] == 'partial':
            self.partials[stream] = event
        elif event['type'] == 'final':
            self.history.append(event)
            self.partials.pop(stream, None)
        elif event['type'] == 'session':
            self.partials.pop(stream, None)
            event['activeStreams'] = len(self.streams)
        for queue in self.subscribers:
            if queue.full():
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait(self.snapshot())
            else:
                queue.put_nowait(event)


class Rooms:
    def __init__(self, ttl=3600, max_rooms=256):
        self.rooms = {}
        self.ttl = ttl
        self.max_rooms = max_rooms

    def prune(self, now=None):
        now = time.monotonic() if now is None else now
        for internal_meeting_id, room in list(self.rooms.items()):
            if not room.streams and not room.subscribers and now - room.updated >= self.ttl:
                del self.rooms[internal_meeting_id]

    def register(self, metadata):
        internal_meeting_id = room_id_from_metadata(metadata)
        self.prune()
        if internal_meeting_id not in self.rooms:
            if len(self.rooms) >= self.max_rooms:
                raise ValueError('Room capacity reached')
            self.rooms[internal_meeting_id] = Room(internal_meeting_id)
        room = self.rooms[internal_meeting_id]
        room.updated = time.monotonic()
        return room

    def resolve(self, command):
        internal_meeting_id = identifier(command.get('internalMeetingId'), 'internalMeetingId')
        self.prune()
        room = self.rooms.get(internal_meeting_id)
        if room is None:
            raise ValueError('Unknown internalMeetingId; BBB must register this room first')
        return room
