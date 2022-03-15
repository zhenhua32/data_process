import hmac
import hashlib

print(hmac.digest(b"key", b"1", digest=hashlib.sha256).hex())
