import redis

print("Starting Redis test...")

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

print("Connected:", r.ping())

r.set("test_key", "hello")
value = r.get("test_key")

print("Value from Redis:", value)
