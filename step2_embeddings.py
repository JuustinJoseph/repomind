import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"

load_dotenv()
client = OpenAI()

SNIPPETS = [
    "def login(username, password):\n    user = db.find_user(username)\n    return bcrypt.check(password, user.hash)",
    "def require_session(request):\n    token = request.cookies.get('session')\n    if not verify_jwt(token): raise Unauthorized()",
    "def resize_image(img, width, height):\n    return img.resize((width, height), Image.LANCZOS)",
    "def charge_card(customer_id, amount_cents):\n    stripe.PaymentIntent.create(customer=customer_id, amount=amount_cents)",
    "def send_welcome_email(user):\n    smtp.send(to=user.email, subject='Welcome!', body=render('welcome.html'))",
]

# Experiment: change to "How do we bill customers?" and watch the ranking flip.
QUESTION = "How does authentication work?"


# How embedding works (notes):
# - The OpenAI server does the work; we just send a list of strings in one API call.
# - Server side: each string is split into tokens (word pieces, which is what usage counts),
#   the tokens pass through a transformer and "read" each other for context (e.g. "password"
#   next to "login"/"bcrypt" shifts toward authentication), then everything is pooled into
#   ONE vector of 1536 numbers (dimensions) per string, scaled to length 1.
# - One vector per string, not per word: 5 strings in -> array of shape (5, 1536).
# - The individual numbers aren't human-readable; meaning lives in the vector's direction,
#   so we compare whole vectors with cosine similarity. Only compare vectors from the same model.
# - Similar meaning -> nearby vectors, even with no shared words ("login" ~ "authentication").
# - Two unrelated topics in one string -> a blended vector between them that matches neither
#   well. Tested: login alone scored 0.359 on the auth question, login+resize scored 0.210.
#   Lesson for step 3: keep chunks focused on one thing (~50 lines).
# - result.data is a list (one item per input, same order); result.data[0].embedding is the floats.
def embed(texts):
    result = client.embeddings.create(model=EMBED_MODEL, input=texts)
    print("Result Of Client Embeddings:")
    print(f"  model: {result.model}")
    print(f"  usage: {result.usage}")  # prompt_tokens / total_tokens (what you pay)
    print(f"  items: {len(result.data)}")  # one per input string
    first = result.data[0]
    print(f"  data[0].index: {first.index}")  # position of that input in your list
    print(
        f"  data[0].embedding: {first.embedding[:5]}... ({len(first.embedding)} floats)"
    )
    np_array = np.array([item.embedding for item in result.data])
    print(f"np_array: {np_array}")
    return np_array


def cosine_similarity(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))


snippet_vecs = embed(SNIPPETS)
question_vec = embed([QUESTION])[0]
print(f"Vector length: {len(question_vec)}")  # expect 1536

scored = [
    (cosine_similarity(vec, question_vec), text)
    for vec, text in zip(snippet_vecs, SNIPPETS)
]
scored.sort(reverse=True)

print(f"\nQuestion: {QUESTION}\n")
for score, text in scored:
    print(f"{score:.3f}  {text.splitlines()[0]}")
