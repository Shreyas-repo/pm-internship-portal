# 🚀 Deployment Guide: GitHub + Vercel

This guide will help you deploy your PM Internship Scheme application to GitHub and then to Vercel for live hosting.

## 📋 Prerequisites

- Git installed on your computer
- GitHub account
- Vercel account (sign up at [vercel.com](https://vercel.com))

## 🔧 Step 1: Prepare Your Project for Deployment

I've already created the necessary deployment files:

- ✅ `.gitignore` - Excludes unnecessary files from Git
- ✅ `vercel.json` - Vercel deployment configuration
- ✅ `runtime.txt` - Specifies Python version
- ✅ Updated `requirements.txt` - Fixed package versions

## 📤 Step 2: Push to GitHub

### 2.1 Initialize Git Repository

Open your terminal/command prompt in the project directory and run:

```bash
cd "C:\Users\Admin\OneDrive\Desktop\job3\job_search"
git init
```

### 2.2 Add Files to Git

```bash
git add .
git commit -m "Initial commit: PM Internship Scheme with ML recommendations"
```

### 2.3 Create GitHub Repository

1. Go to [GitHub.com](https://github.com) and sign in
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Fill in the details:
   - **Repository name**: `pm-internship-portal`
   - **Description**: `AI-powered internship portal with ML recommendations`
   - **Visibility**: Public (or Private if you prefer)
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)

### 2.4 Connect Local Repository to GitHub

Replace `YOUR_USERNAME` with your actual GitHub username:

```bash
git remote add origin https://github.com/YOUR_USERNAME/pm-internship-portal.git
git branch -M main
git push -u origin main
```

## 🌐 Step 3: Deploy to Vercel

### 3.1 Connect GitHub to Vercel

1. Go to [vercel.com](https://vercel.com) and sign in
2. Click "New Project"
3. Import your GitHub repository:
   - Select "Import Git Repository"
   - Choose your `pm-internship-portal` repository
   - Click "Import"

### 3.2 Configure Deployment Settings

Vercel will automatically detect it's a Python project. Configure these settings:

**Build & Development Settings:**
- **Framework Preset**: Other
- **Build Command**: Leave empty (Vercel handles this)
- **Output Directory**: Leave empty
- **Install Command**: `pip install -r requirements.txt`

### 3.3 Set Environment Variables

In the Vercel dashboard, add these environment variables:

```
SECRET_KEY=your-super-secret-key-here-make-it-long-and-random
FLASK_ENV=production
```

To generate a secure secret key, you can use:
```python
import secrets
print(secrets.token_hex(32))
```

### 3.4 Deploy

1. Click "Deploy"
2. Wait for the build to complete (usually 2-3 minutes)
3. Your app will be live at `https://your-project-name.vercel.app`

## 🔄 Step 4: Automatic Deployments

Once connected, Vercel will automatically deploy:
- Every push to the `main` branch triggers a production deployment
- Pull requests create preview deployments

## 🐛 Common Issues & Solutions

### Issue 1: ML Model File Too Large
If your `.pkl` file is too large for Vercel:

**Solution**: Train the model on first deployment
```python
# In app.py, add this code to train model if not found
if not os.path.exists(model_path):
    print("Training ML model on first deployment...")
    from ml_model.internship_reccomander import train_and_save_model
    train_and_save_model(csv_path='ml_model/Internship.csv', model_path=model_path)
```

### Issue 2: Database Issues
Vercel's serverless functions are stateless, so SQLite might not persist.

**Solution**: For production, consider using:
- **PostgreSQL** (recommended): Vercel Postgres or external service
- **MySQL**: PlanetScale or external service

### Issue 3: Cold Starts
First request might be slow due to serverless cold starts.

**Solution**: This is normal for free tier. Consider Vercel Pro for faster cold starts.

## 📊 Step 5: Monitor Your Deployment

### Vercel Dashboard Features:
- **Functions**: Monitor serverless function performance
- **Analytics**: Track page views and performance
- **Logs**: Debug issues with real-time logs
- **Domains**: Add custom domain names

### Useful Commands for Updates:

```bash
# Make changes to your code
git add .
git commit -m "Description of changes"
git push origin main
# Vercel automatically deploys the changes
```

## 🔒 Security Considerations

1. **Environment Variables**: Never commit sensitive data to Git
2. **Secret Key**: Use a strong, unique secret key
3. **Database**: Consider using environment-specific databases
4. **HTTPS**: Vercel provides HTTPS by default

## 📈 Performance Optimization

1. **Static Files**: Vercel serves static files from CDN automatically
2. **Caching**: Add appropriate cache headers for better performance
3. **Database**: Use connection pooling for production databases
4. **ML Model**: Consider caching model predictions

## 🎯 Next Steps After Deployment

1. **Custom Domain**: Add your own domain in Vercel settings
2. **Analytics**: Set up Vercel Analytics for insights
3. **Monitoring**: Add error tracking (Sentry, etc.)
4. **Database**: Migrate to a production database service
5. **CI/CD**: Set up automated testing with GitHub Actions

## 📞 Support

If you encounter issues:
- Check Vercel deployment logs
- Review GitHub repository settings
- Consult Vercel documentation
- Check this project's issues on GitHub

---

**Your app will be live at**: `https://your-project-name.vercel.app`

🎉 **Congratulations!** Your PM Internship Scheme portal is now live and accessible worldwide!
