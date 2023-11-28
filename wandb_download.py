import wandb

# Authenticate (you'll be prompted to enter your API key)
wandb.login()

# Initialize the API
api = wandb.Api()

# Fetch a specific run
run = api.run("airlabdepth/dsta-mvs-sweep/t83e0bck")



# Download images from the 'media/images' folder
for file in run.files():
    if "media/images" in file.path and file.name.endswith('.png'):  # Adjust the file format if needed
        file.download(root='/home/migo/Downloads/depth_estimation/dsta_inference_gascola')  # Specify your local download directory
