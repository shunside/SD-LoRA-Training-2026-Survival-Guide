# SD LoRA Training: 2026 Survival Guide

It’s 2026, and the 2023 guide I found on the first page of DuckDuckGo doesn't exactly "just work" anymore. Package dependencies have a habit of falling apart over time, and I got tired of watching Colab throw errors every time I tried to train something. 

I’ve spent some time fixing the dependency hell in the `Lora_Trainer.ipynb` notebook so it actually runs without throwing a tantrum. This is a working summary of the original workflow for anyone who just wants to train a LoRA without debugging library conflicts for hours—or minutes with an AI, idk.

I’m basing this on the original guide, so keeping it open in a side tab for extra context is a good idea. However, I’m not looking to fuel your tab-hoarding habit; I’ll point out the specific moments when you actually ***must*** open it so you can keep your RAM usage (and sanity) under control.

#### 🔗 The Essentials

| Resource | Description | Link |
| :--- | :--- | :--- |
| **Original Author** | Hollowstrawberry's profile | [GitHub](https://github.com/hollowstrawberry/) |
| **Dataset Maker** | Original version (Still functional, I guess) | [Dataset_Maker.ipynb](https://colab.research.google.com/github/hollowstrawberry/kohya-colab/blob/main/Dataset_Maker.ipynb) |
| **LoRA Trainer (SD1.5)** | Original (The one that currently not well working) | [Lora_Trainer.ipynb](https://colab.research.google.com/github/hollowstrawberry/kohya-colab/blob/main/Lora_Trainer.ipynb) |
| **LoRA Trainer (SDXL)** | Original (For XL users) | [Lora_Trainer_XL.ipynb](https://colab.research.google.com/github/hollowstrawberry/kohya-colab/blob/main/Lora_Trainer_XL.ipynb) |
| **My Fixed Version** | **The one that actually runs for now (SD1.5)** | [Lora_Trainer.ipynb](https://colab.research.google.com/github/shunside/LoraTrainer/blob/main/Lora_Trainer.ipynb) |
| **Inference** | For those without an NVIDIA GPU like me (Automatic1111) | [fast_stable_diffusion](https://colab.research.google.com/github/TheLastBen/fast-stable-diffusion/blob/main/fast_stable_diffusion_AUTOMATIC1111.ipynb) |

#### 🔧 What’s what?

A quick breakdown of the resources above, so you don't have to click everything to figure out what they do:

* **Original Author:** Credit goes to **Hollowstrawberry**. While the entire ecosystem isn't solely theirs, their guide is the foundation I’m building on. If you read the original article, you'll see why.
* **Dataset Maker:** This is the factory where your training data is born. 
    * **The Gist:** You need a collection of images and matching descriptions (e.g., `1.png` and `1.txt`). 
    * **Honesty Corner:** I only ever used **Step 1** (Setup) and **Step 6** (Ready). I have no idea if the scraping or tagging steps in between actually work in 2026, nor did I touch any "extra" functions. Use them at your own risk or benefit.
* **LoRA Trainer (SD1.5 vs SDXL):** * Step 6 of the Dataset Maker usually just yeets you into the SD1.5 trainer. 
    * If you're wondering which one to use: **SD 1.5** is generally more "okay" for generating realistic characters (my personal preference for this guide), while **SDXL** is the beefier, high-res sibling. I haven't even bothered looking at the SDXL notebook, so you're on your own there.
* **My Fixed Version (The 2026 Patch):** This is the SD1.5 trainer, but cleaned up. 
    * **Dependency Fix:** No more library version tantrums at the start of the script. 
    * **Subfolder Support:** The original expects all images and `.txt` files to be dumped directly into the `dataset` folder. I modified it to support subfolders. You can now keep things organized (e.g., `/dataset/face`, `/dataset/full_body`, `/dataset/back`) without the script breaking. It’s a small tweak, but if you’re manually sorting data like me, it’s a lifesaver.
* **Inference (The "I don't have an NVIDIA card" option):** Once you get your `.safetensors` file, you need to test it. Since I don't have the budget for a high-end NVIDIA GPU either, I included this link. It lets you run your model through industry-standard methods (like Automatic1111) using the **Nano Banana Pro** model (which is arguably the best production model out there right now) for free.

---

#### 🗺️ The Roadmap: How I did it (The Manual Way)

You can use the original author's automated steps (Gelbooru scraping, FiftyOne AI cleaning, and BLIP tagging). It’s a cool "one-click" dream, but if you want a LoRA that actually looks like a specific person/character, automation usually throws a tantrum. 

Here is the manual, high-quality path I followed for 2026:

##### 1. The Setup (Dataset Maker)
Open the **Dataset Maker** Colab and run **Step 1**. 
* **Crucial Choice:** Set `folder_structure` to `Organize by project (MyDrive/Loras/project_name/dataset)`. 
* **Why?** It’s cleaner, it looks better in your Drive, and it’s the most stable path for the next steps. Just do it.

##### 2. Crafting the Visual DNA
I didn't just scrape random images. I built a specific character from scratch.
* **Subfolders:** Inside your `dataset` folder, create subfolders like `/face`, `/full_body`, `/waist`, and `/back`. This is where my **Fixed Version** shines by supporting these paths.
* **The Face Strategy:** I used **Gemini Nano Banana Pro** (standalone chat) to generate a base face. After trying face-swapping (which failed about 60% of the time), I switched tactics: I used the base face as a reference and asked the AI for variations in different lighting, camera angles, and lenses.
* **The Rule of 20:** You need about 15-20 high-quality face shots for a LoRA to go from "meh" to "actually good."

##### 3. Gathering the Body & Environment
I filled the rest of the dataset (~106 images total) using Pinterest and other sources to get the body types and vibes I wanted. 

##### 4. The Tagging Grind (The "Nerd" Part)
Automation is fast, but manual tagging is king. I named every file `1.png`, `2.png`... and created matching `1.txt`, `2.txt` files. To get the perfect tags, I used this specific prompt with Gemini:

<details>
<summary><b>Click to expand: The Ultimate Tagger Prompt</b></summary>

```text
Act as an expert AI Vision Tagger specializing in training LoRA models for Stable Diffusion (Kohya_ss/SDXL).

I am creating a dataset for a specific character named "{ur_character_name}".
Her visual DNA is: {Describe visual DNA here}

I will upload images to you one by one. Your goal is to provide the perfect "Caption Tags" for each image.

Follow these strict tagging rules:
1. THE TRIGGER WORD RULE: ALWAYS start every caption with: "{ur_character_name}, woman, ..."
2. THE "FACE" LOGIC: Tag facial features if visible. If not visible, you MUST add "headless" or "no face". If obscured, use "phone/camera covering face".
3. BODY & OUTFIT: Describe visible parts explicitly (stomach, cleavage, yoga pants, etc.).
4. FORMATTING: Output ONLY tags separated by commas. No sentences.

Format: "{ur_character_name}, woman, [framing], [body/face], [clothing], [pose], [background]"
```
</details>

I fed the images to Gemini (sometimes one by one, sometimes in batches when I got bored) and copy-pasted the tags into the `.txt` files.

##### 5. Training (The Magic Moment)
Once the dataset was ready, I moved to My Fixed Version of the LoRA Trainer.
* **The "Settings Wall"**: The first cell has a lot of values. If you feel overwhelmed, ask an AI to explain the hyperparameters.
* **The Wait:** Training usually takes about 15-30 minutes in the 2026 Colab environment.

##### 6. The Prize
After training, check your Google Drive: `{project_name}/output/`. You’ll find several `.safetensors` files. Usually, the one with the highest number is the most "evolved" version of your model.

---

#### 🧪 Inference: Testing your LoRA without a GPU

If you don't have a high-end NVIDIA card (and I don't either), we have to rely on Google’s GPU charity. It’s 2026, and Colab still acts like a stingy landlord with its resources, but it's the only way to run industry-standard interfaces like Automatic1111 for free.

##### 1. Begging for a GPU
Before running anything, you need to tell Colab you aren't trying to mine Bitcoin on a CPU. 
1. Go to the top menu: **Runtime** -> **Change runtime type**.
2. Select **T4 GPU** (or whatever the best free option is today). 
3. Click **Save**.

If you see a popup saying *"Cannot connect to GPU backend"* or asking you to *"Pay As You Go,"* it means the free limit is hit. You *can* continue with a CPU, but don't. It won't work for what we’re doing. Just wait for your limit to reset or try another account.

##### 2. The Setup Phase
* **Cell 1 (Mount Drive):** Leave `Shared_Drive` empty. Run the cell and wait for the green "Done" checkmark.
* **Cells 2 & 3:** Run these sequentially once the previous ones finish.

##### 3. Model Configuration (Cell 4)
* **Use_Temp_Storage:** Uncheck this if you have enough space in your Google Drive. If you plan on using this regularly, keep it unchecked.
* **Model_Version:** Since we trained an SD 1.5 model, select **v1.5**. If you went the XL route, choose SDXL.
* **PATH_to_MODEL / MODEL_LINK:** Leave these empty.

##### 4. The LoRA Link Saga
Now you need to import your trained `.safetensors` file. This is usually where things go south. 

1. Go to your Google Drive, find your LoRA file in `{project_name}/output/`.
2. Right-click -> **Share** -> Set access to **"Anyone with the link"** and copy the link.
3. It will look like this: `https://drive.google.com/file/d/[FILE_ID]/view?usp=sharing`
4. Paste this into the `LoRA_LINK` field and run the cell.

**If it fails (The "Unlucky Desert Person" Scenario):**
Direct Drive links often break. If it doesn't download, try using this direct download schema:
`https://drive.google.com/uc?id=[YOUR_FILE_ID_HERE]`

**The Last Resort (Manual Move):**
If the cell still refuses to cooperate, don't waste time debugging it. Copy your `.safetensors` file manually in your Google Drive. Based on the notebook's internal logic, you need to paste it here:
`MyDrive/sd/stable-diffusion-webui/models/Lora/`

Once the file is there, you can skip the Download LoRA cell and move on.

##### 5. ControlNet (Optional)
You'll see a ControlNet cell. Unless you have a burning desire to max out your VRAM and crash the session, **leave everything as "None."** ControlNet on a free Colab instance is a gamble that usually isn't worth the lag. If you must, keep it minimal.

##### 6. Start Stable-Diffusion
* **Ngrok_token:** Leave empty unless you specifically need a persistent external link.
* **User / Password:** Leave empty if you are the only one using it. 
Run the cell. It will generate a public Gradio link. Open it, and you’re in. 

From here, you can finally load your LoRA in the "Additional Networks" or "LoRA" tab and see if those 20 face photos actually paid off.
