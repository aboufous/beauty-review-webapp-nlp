COSC3801/3015 Assignment 3 Milestone II - HD Web App Skeleton

## Team Setup

Clone repo:

```bash
git clone

How to run:
1. Create a virtual environment
   python -m venv .venv
   source .venv/bin/activate   # Mac/Linux
   .venv\Scripts\activate      # Windows

2. Install dependencies
   pip install -r requirements.txt

3. Run the app
   streamlit run app.py

What this skeleton includes:
- Task 1: Product browsing and fuzzy keyword search by brand/name/description.
- Task 2: Review creation form with predicted recommendation label and user override.
- Task 3: Similar item recommendation using TF-IDF cosine similarity.
- Task 4: Additional admin analytics dashboard for review sentiment/label/rating patterns.
- HD extension points: fused prediction architecture using three evidence sources:
  1. Review text model
  2. Rating/meta model
  3. Product-history prior model

Replace the sample data in data/products.csv with the Milestone I dataset.
Replace models/model_loader.py placeholders with your trained Milestone I models.

Group members:
- Add student names and IDs here.

Video demo checklist:
1. Browse products and search with spelling variation.
2. Open product detail page.
3. Create a review and show generated label.
4. Override label and save review.
5. Show similar item recommendations.
6. Show admin analytics dashboard.
