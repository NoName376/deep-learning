import cv2
import easyocr
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

image_path = r"midterm.jpg"

ground_truth = """Kazakh-British Technical University · School of IT and Engineering
CSCI1409 Deep Learning · Fall 2026–2027

Midterm Examination

Week 8 attestation · Covers Weeks 1–7 · 40 questions + 1 bonus · 10 points · 120 minutes

Full name: Bakhramov Abdallahi
Student ID: 24B031692
Group:
Date:

Instructions
• Section A (Questions 1–10): multiple choice, 0.15 points each. Circle one letter for each question.
• Section B (Questions 11–20): calculations, 0.35 points each. Show your working in the box — a correct method earns partial credit even if the final number is wrong. Round to 3 decimal places; ln is the natural logarithm.
• Section C (Questions 21–40): open questions, 0.25 points each. Answer concisely in the lined space. Bullet points, equations and corrected code are acceptable; there is no minimum length.
• Bonus (Question 41): optional, +0.25 points on top of the 10-point total (maximum score 10.25).
• Short, precise answers can earn full marks. If you get stuck on a question, move on and come back to it later.
• Closed book. A non-programmable calculator is allowed. Phones, notes, laptops, smartwatches and AI tools are not allowed.

Academic integrity
Copying, using unauthorised materials or devices, or communicating with other students during the exam is a violation of KBTU academic policy and will be reported. All answers must be your own work.

For Instructor use
Section | Questions | Points | Score
A — Multiple choice | 1–10 | 1.50 |
B — Calculations | 11–20 | 3.50 |
C — Open questions | 21–40 | 5.00 |
Total | 40 | 10.00 |
Bonus (optional) | 41 | +0.25 |

6.05
Page 1 of 21""".strip()


def calculate_cer(ref, hyp):
    if not ref:
        return 0.0 if not hyp else 1.0
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost
            )
    return dp[m][n] / len(ref)


print("Running Tesseract Model")
img_pil = Image.open(image_path)

tesseract_text = pytesseract.image_to_string(img_pil, lang="eng")

print("Result Tesseract:")
print(tesseract_text)

tess_cer = calculate_cer(ground_truth, tesseract_text.strip())
print(f"Tesseract CER: {tess_cer}\n")


print("\n" * 5)


print("Running EasyOCR Model")

reader = easyocr.Reader(["en"], gpu=True)

easyocr_results = reader.readtext(image_path)

print("Result EasyOCR:")
easyocr_text_full = []
for bbox, text, prob in easyocr_results:
    print(f"[{prob:.2f}] {text}")
    easyocr_text_full.append(text)

easy_text = " ".join(easyocr_text_full).strip()
easy_cer = calculate_cer(ground_truth, easy_text)
print(f"EasyOCR CER: {easy_cer}")