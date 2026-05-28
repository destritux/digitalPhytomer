import random
import string

class BlackBoxAPI:
    """
    A simulated challenge-response API endpoint for reverse engineering testing.
    The goal of the agents is to extract the secret access key.
    To do so, they must submit the correct encoded payload matching a randomized challenge.
    """
    def __init__(self, seed=42):
        self.secret_key = "FLAG{H0L4RCHY_AD4PTAT10N_2026}"
        self.seed = seed
        random.seed(seed)
        
    def generate_challenge(self):
        """
        Generates a random 6-character alphabetic challenge string.
        """
        return "".join(random.choices(string.ascii_letters, k=6))
        
    def encode_correctly(self, text):
        """
        The hidden mathematical/cryptographic transformation logic:
        1. Shift each character by its index in the string + a fixed offset of 7 (modulo 256).
        2. Reverse the resulting string.
        """
        shifted = []
        for idx, char in enumerate(text):
            offset = 7 + idx
            new_char_code = (ord(char) + offset) % 256
            shifted.append(chr(new_char_code))
        return "".join(shifted)[::-1]
        
    def submit(self, challenge, payload):
        """
        Accepts a challenge and a payload.
        If the payload matches the correct encoding, returns the access key.
        If not, returns an error along with a helpful hint explaining how the submitted challenge is encoded.
        """
        expected = self.encode_correctly(challenge)
        if payload.strip() == expected:
            return {
                "success": True,
                "message": "Access Granted",
                "key": self.secret_key
            }
        else:
            # Dynamic hint for the actual challenge submitted
            ref_encoded = self.encode_correctly(challenge)
            return {
                "success": False,
                "message": "Error: Invalid payload encoding.",
                "hint": f"For reference challenge '{challenge}', the expected correctly encoded payload is '{ref_encoded}'."
            }

if __name__ == "__main__":
    # Test verify
    api = BlackBoxAPI()
    ch = api.generate_challenge()
    correct = api.encode_correctly(ch)
    print(f"Challenge: {ch}")
    print(f"Correct Payload: {correct}")
    res = api.submit(ch, correct)
    print(res)
    
    # Test failure hint
    bad = api.submit(ch, "wrong_payload")
    print(bad)
