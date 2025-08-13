# Deployment Guide

This guide provides step-by-step instructions on how to deploy this Flask application to a production environment. We will use **Render**, a modern cloud platform that offers a generous free tier for web services.

## Prerequisites

- A GitHub account with this project's code pushed to a repository.
- A free account on [Render](https://render.com/).

## Step 1: Prepare the Application for Production

A production environment requires a more robust web server than the one that comes with Flask's development server. We will use **Gunicorn**.

### 1a. Add Gunicorn to `requirements.txt`

Make sure the `requirements.txt` file includes `gunicorn`. The file should look something like this:

```
Flask
pandas
numpy
scikit-learn
plotly
requests
openpyxl
gunicorn
```

### 1b. Create a `Procfile`

The `Procfile` is a simple text file that tells the deployment platform how to run the application. Create a new file named `Procfile` (with no extension) in the root of your project and add the following line:

```
web: gunicorn app:app
```

This command tells Render to start a web process by running the `gunicorn` server and pointing it to the `app` object inside your `app.py` file.

## Step 2: Deploy on Render

Render can deploy your application directly from your GitHub repository.

### 2a. Create a New Web Service

1.  Log in to your Render account.
2.  On the Dashboard, click **"New +"** and select **"Web Service"**.
3.  Connect your GitHub account and select the repository for this project.
4.  On the settings page, configure your service:
    - **Name:** Give your application a unique name (e.g., `my-datasci-pipeline`).
    - **Region:** Choose a region close to you.
    - **Branch:** Select the main branch of your repository.
    - **Root Directory:** Leave this as is if your `app.py` is in the root directory.
    - **Runtime:** Select **"Python 3"**.
    - **Build Command:** `pip install -r requirements.txt` (this is usually the default).
    - **Start Command:** `gunicorn app:app` (Render will automatically use the `Procfile` if it exists, but you can also enter it here).
    - **Instance Type:** The **"Free"** tier is sufficient for this application.

### 2b. Add an Environment Variable (if needed for secret key)

For better security, it's good practice to not hardcode the `app.secret_key`. If you were to change `app.secret_key = os.environ.get('SECRET_KEY')` in `app.py`, you would add the environment variable here:

1.  Click on the **"Environment"** tab for your new service.
2.  Click **"Add Environment Variable"**.
3.  **Key:** `SECRET_KEY`
4.  **Value:** Generate a long, random string and paste it here.

### 2c. Deploy

1.  Scroll to the bottom of the page and click **"Create Web Service"**.
2.  Render will automatically pull your code from GitHub, install the dependencies from `requirements.txt`, and start the application using the Gunicorn command.
3.  You can monitor the deployment process in the "Events" or "Logs" tab.
4.  Once the deployment is complete, Render will provide you with a public URL (e.g., `https://my-datasci-pipeline.onrender.com`) where you can access your live application.

## Automatic Deploys

By default, Render will set up a webhook with your GitHub repository. This means that every time you push a new commit to your main branch, Render will automatically re-deploy the application with the latest changes.
