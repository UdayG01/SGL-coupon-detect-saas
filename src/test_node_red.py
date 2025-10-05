import time
import requests
from gpiozero import LED, Device
from gpiozero.pins.mock import MockFactory

# Mock GPIO
Device.pin_factory = MockFactory()
ok_led = LED(17)
ng_led = LED(27)

while True:
    signal = input("> ")
    if signal.upper() == "OK":
        print("OK triggered (GPIO17 HIGH)")
        ok_led.on()
        requests.post("http://localhost:1880/ok", json={"pin": 17, "state": 1})
        time.sleep(1)
        ok_led.off()
        requests.post("http://localhost:1880/ok", json={"pin": 17, "state": 0})
    elif signal.upper() == "NG":
        print("NG triggered (GPIO27 HIGH)")
        ng_led.on()
        requests.post("http://localhost:1880/ng", json={"pin": 27, "state": 1})
        time.sleep(1)
        ng_led.off()
        requests.post("http://localhost:1880/ng", json={"pin": 27, "state": 0})
    else:
        print("Invalid input. Type OK or NG.")
