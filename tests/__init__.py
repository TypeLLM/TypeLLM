import os

# A developer's key would send clients built without base_url to the hosted API.
os.environ.pop("TYPELLM_API_KEY", None)
