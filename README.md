# NEST Property Investment Studio

A Streamlit dashboard for exploring Indian housing listings, estimating a property's asking price with a regression model, classifying its relative price tier, and testing investor-provided budget and rental assumptions.

## Project files

```text
app.py
requirements.txt
india_housing_prices.csv
assets/hero.svg
.streamlit/config.toml
```

Keep the CSV beside `app.py` and preserve the dataset's column names. The included SVG is the local hero illustration; the app does not need a remote image service.

## Add to GitHub

Copy this whole folder's contents into the root of your GitHub repository. The dataset is about 41 MB: GitHub's browser upload has a 25 MiB per-file limit, so use GitHub Desktop or Git from the repository folder instead. A regular Git push supports this file size.

```powershell
git add app.py requirements.txt india_housing_prices.csv assets .streamlit README.md
git commit -m "Add NEST property investment app"
git push
```

## Deploy on Streamlit Community Cloud

1. Sign in at [share.streamlit.io](https://share.streamlit.io/) with GitHub.
2. Choose **Create app** and select the repository and branch.
3. Set the main file path to `app.py`.
4. Use Python 3.12 in Advanced settings, then deploy.
5. Open the `*.streamlit.app` link after the build completes.

The app's models are trained at runtime and cached. Regression estimates asking price in lakhs. Classification predicts a relative Value, Mid-market, or Premium price tier. Metrics are evaluated on a held-out split. The app's return scenario is an illustration based on user-entered rent, vacancy, costs, and appreciation assumptions; it is not financial advice.

## Adding or updating data

Use **Data workspace** to add a comparable listing. It immediately joins the charts and the next model training sample for the current Streamlit session. Use **Download full updated CSV** to keep the new row. To make it permanent for all users, replace the repository CSV with the downloaded file and push the change to GitHub. Alternatively, upload a CSV in the sidebar to analyze it for that session.

Streamlit Cloud sessions are temporary, so app-entered values are not permanently saved by the server. This download-and-commit step is required for durable updates.
