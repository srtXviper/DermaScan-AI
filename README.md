# DermaScan AI complete package

This package replaces the old Transformer backend with your trained ResNet18 checkpoint.

## Before deploying
Your public Hugging Face repository `SRTxVIPER/dermascan-skin-cnn` must contain `skin_cnn.pth` at the repository root. It must be the checkpoint created by your Colab training code and contain `state_dict`, `class_names`, and `num_classes`.

## Render deployment
1. Download and extract this ZIP.
2. Back up your existing GitHub repository.
3. Upload the CONTENTS of this folder to the repository root, preserving the `static` folder.
4. Commit the changes to the branch connected to Render.
5. In Render, use the repository root as Root Directory.
6. Build command: `pip install --upgrade pip && pip install -r requirements.txt`
7. Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
8. Set Python version to 3.11.11 if Render asks.
9. Redeploy. Then open `/api/model-info` to test the checkpoint download and loading, followed by `/` to test the website.

The included `render.yaml` contains these settings. The app uses `HF_REPO_ID` and `MODEL_FILENAME`; defaults are already configured.

## Important
No one can guarantee a hosted deployment has zero errors without running it in the target account. Render free instances can have memory/cold-start limits for PyTorch. If Render runs out of memory, the hosting plan/platform must be changed. The model is experimental, not clinically validated, and must not be used for diagnosis or treatment.
