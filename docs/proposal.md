# Text-to-Image Generation with Stable Diffusion and CLIP

**Team:** Gokul Chandar Kumaran, Loalith Madala, Yagna Praneeth Reddy Panguru · **Date:** November 2026
**Repo:** [github.com/GokulChandarKumaran/text2image-diffusion](https://github.com/GokulChandarKumaran/text2image-diffusion)

## 1. What we want to do
Text-to-image models are impressive, but they still trip up on fairly ordinary requests. Ask for "two cats on a couch next to a remote" and the remote often just disappears. Our project looks at how closely a pretrained diffusion model follows its prompt on everyday scenes, and whether we can tighten that up by fine-tuning it on a small, carefully cleaned set of captioned photos. We'll count it as a success if the fine-tuned model beats the off-the-shelf version on CLIPScore and retrieval accuracy, and if people prefer its images in a blind comparison.

## 2. Dataset
We're using the **MS-COCO 2017 validation set**: 5,000 photos of everyday scenes, each with five captions written by people (about 25,000 in total). We picked COCO over Flickr30k for two reasons. It covers far more kinds of objects and scenes (80 categories), and it's the dataset most text-to-image papers report on, so our numbers can be compared with theirs. The validation split is also just the right size for this milestone: about 1 GB, compared with 18 GB for the full training set.

Raw COCO captions are a little messy, so we cleaned them before using them. We tidied up spacing and punctuation, removed duplicates, and kept only captions between 5 and 40 words that fit within CLIP's 77-token limit. On the image side, we dropped photos that were too small (under 256 px) or too stretched (wider than 2:1), then resized and cropped the rest to 512×512. That left us with **4,926 image–caption pairs**: 4,434 for training and 492 held out for testing. For each image we use its best caption and keep the others for data augmentation later.

## 3. Model and approach
We're building on **Stable Diffusion v1.5**, which uses **CLIP** (ViT-L/14) to turn each prompt into the numbers the image model works from. It runs comfortably on the 8 GB laptop GPUs we have.

- **Step 1, baseline (done):** we measured how well things work before changing anything. CLIP picks the correct photo out of 492 for a given caption 67% of the time, and the correct photo lands in its top 5 picks 91% of the time. Our five test generations took about 5 seconds each and scored 28.9 on CLIPScore. They also showed the kinds of mistakes we want to fix: a "double-decker" bus with only one deck, and a remote control that never showed up.
- **Step 2, fine-tuning:** we'll train the model on our cleaned captions using LoRA, which updates only a small part of the model (about 3 million parameters), so it fits on our hardware.
- **Step 3, experiments:** we'll compare different settings (how strictly the model follows the prompt, how many generation steps it takes, which caption we train on) to see what actually makes a difference.
- **How we'll judge it:** CLIPScore, FID on the 492 held-out images, retrieval accuracy, and a small blind survey where people pick the image that matches better.

## 4. Who's doing what
| Member | Role | Responsible for |
|---|---|---|
| Gokul Chandar Kumaran | Modeling & integration | Generation pipeline, fine-tuning, GPU setup, the repo, milestone reports |
| Loalith Madala | Data | Downloading, cleaning and filtering, dataset analysis, caption augmentation |
| Yagna Praneeth Reddy Panguru | Evaluation | Scoring scripts (CLIPScore, FID, retrieval), human survey, result plots, demo |

We'll all work on the final write-up and presentation together.

## 5. Where we are, and what could go wrong
Everything for Milestone 1 is in the repo: the data cleaning pipeline, the environment setup, the baseline notebook, and our first five generated images. The main risk is GPU memory. We're handling it by running the model at half precision and using LoRA rather than retraining the whole model. COCO captions can also be noisy, which is why we filtered them and picked the best one for each image. Finally, COCO images come with their own license terms, so we're using them for research only.
