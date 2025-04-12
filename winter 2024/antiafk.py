import pydirectinput
import time

def send_inputs():
    while True:
        # Press and hold 'a' for 2 seconds
        pydirectinput.keyDown('a')
        time.sleep(2)
        pydirectinput.keyUp('a')

        # Wait 10 seconds
        time.sleep(10)

        # Press and hold 'd' for 2 seconds
        pydirectinput.keyDown('d')
        time.sleep(2)
        pydirectinput.keyUp('d')

        # Wait 10 seconds
        time.sleep(10)

if __name__ == "__main__":
    print("Starting to send 'a' and 'd' inputs every 10 seconds...")
    try:
        send_inputs()
    except KeyboardInterrupt:
        print("\nStopped by user.")
