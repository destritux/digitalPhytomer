import time

class TelemetryManager:
    events_log = []

    @classmethod
    def log_event(cls, event_type: str, data: dict):
        cls.events_log.append({
            "timestamp": time.time(),
            "event": event_type,
            "data": data
        })

    @classmethod
    def print_report(cls):
        print("\n" + "="*40)
        print("SYSTEM COGNITIVE TELEMETRY REPORT")
        print("="*40)
        counts = {}
        for ev in cls.events_log:
            name = ev["event"]
            counts[name] = counts.get(name, 0) + 1
        
        for k, v in sorted(counts.items()):
            print(f"- Event '{k}': triggered {v} times.")
        print("="*40 + "\n")

    @classmethod
    def get_event_count(cls, event_type: str) -> int:
        return sum(1 for ev in cls.events_log if ev["event"] == event_type)

class CognitiveEventBus:
    def __init__(self):
        self.subscribers = {}

    def subscribe(self, event_type: str, callback):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)

    def emit(self, event_type: str, data: dict):
        # Always log to telemetry
        TelemetryManager.log_event(event_type, data)
        
        if event_type in self.subscribers:
            for cb in self.subscribers[event_type]:
                try:
                    cb(data)
                except Exception as e:
                    print(f"[EventBus Error] Callback failed for event '{event_type}': {e}")
