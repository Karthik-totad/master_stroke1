"""
core/mqtt_publisher.py

Single shared MQTTPublisher for the entire NeuroRehab system.
Used by both fusion and recovery features.
"""

import json
import time
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False


class MQTTPublisher:
    """Shared MQTT publisher for fusion and recovery data."""

    def __init__(self, broker, port, patient_id, max_retries=3):
        self.pid = patient_id
        self.broker = broker
        self.port = port
        self.connected = False
        self.client = None
        self._reconnect_attempts = 0
        self._max_reconnect = 5

        if not MQTT_AVAILABLE:
            print("[MQTT] paho-mqtt not installed — publishing disabled.")
            return

        self._create_client()
        self._connect_with_retries(max_retries)

    def _create_client(self):
        """Create MQTT client with callbacks."""
        self.client = mqtt.Client(
            client_id=f"neurorehab_{self.pid}_{int(time.time())}"
        )
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish = self._on_publish

    def _connect_with_retries(self, max_retries):
        """Connect with retry logic."""
        for attempt in range(max_retries):
            try:
                print(f"[MQTT] Connecting to {self.broker}:{self.port} (attempt {attempt+1}/{max_retries})...")
                self.client.connect(self.broker, self.port, keepalive=60)
                self.client.loop_start()
                # Wait for connection to establish
                time.sleep(0.5)
                if self.connected:
                    print(f"[MQTT] Connected successfully on attempt {attempt+1}")
                    self._reconnect_attempts = 0
                    break
                else:
                    print(f"[MQTT] Connection attempt {attempt+1} did not complete yet, waiting...")
                    time.sleep(0.5)
            except Exception as e:
                print(f"[MQTT] Connection attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.5)
        else:
            print(f"[MQTT] Could not connect after {max_retries} attempts. Will retry on publish.")

    def _on_connect(self, client, userdata, flags, rc):
        self.connected = True
        self._reconnect_attempts = 0
        print(f"[MQTT] Connected (rc={rc})")

    def _on_disconnect(self, client, userdata, rc):
        self.connected = False
        print(f"[MQTT] Disconnected (rc={rc}) - will attempt reconnect")

    def _on_publish(self, client, userdata, mid):
        print(f"[MQTT] Published message (mid={mid})")

    def _ensure_connected(self):
        """Ensure connection is alive, reconnect if needed."""
        if not MQTT_AVAILABLE:
            return False
        if not self.client:
            self._create_client()
        if not self.connected:
            if self._reconnect_attempts < self._max_reconnect:
                self._reconnect_attempts += 1
                print(f"[MQTT] Reconnecting (attempt {self._reconnect_attempts}/{self._max_reconnect})...")
                try:
                    self.client.loop_stop()
                    self._connect_with_retries(2)
                except Exception as e:
                    print(f"[MQTT] Reconnect failed: {e}")
            else:
                print(f"[MQTT] Max reconnect attempts reached")
                return False
        return self.connected

    def publish(self, subtopic, payload_dict):
        """Publish to rehab/{patient_id}/{subtopic}"""
        if not MQTT_AVAILABLE:
            return

        # Ensure we're connected before publishing
        if not self._ensure_connected():
            print(f"[MQTT] SKIP {subtopic}: not connected")
            return

        topic = f"rehab/{self.pid}/{subtopic}"
        try:
            result = self.client.publish(topic, json.dumps(payload_dict), qos=0)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"[MQTT] PUBLISH {topic}")
            else:
                print(f"[MQTT] PUBLISH failed: rc={result.rc}")
                self.connected = False
        except Exception as e:
            print(f"[MQTT] Publish error: {e}")
            self.connected = False
