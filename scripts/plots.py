import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# # Load the dataset (replace 'your_file.csv' with the actual file path)
# df = pd.read_csv("/homes/gws/yifangc/verl/scripts/test.csv")

# # Convert "null" to NaN for proper handling
# df.replace("null", np.nan, inplace=True)

# # Convert columns to numeric (handle potential parsing errors)
# df = df.apply(pd.to_numeric, errors='coerce')

# # Drop rows with NaN values
# df.dropna(inplace=True)

# print(f"Size of dataset: {df.shape}")



# # Scatter plots to visualize relationships
# fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# sns.scatterplot(x=df['epoch0_meanR'], y=df['epoch0_var'], ax=axes[0, 0])
# axes[0, 0].set_title('Epoch 0: Variance vs. Reward')

# sns.scatterplot(x=df['epoch1_meanR'], y=df['epoch1_var'], ax=axes[0, 1])
# axes[0, 1].set_title('Epoch 1: Variance vs. Reward')

# sns.scatterplot(x=df['epoch0_meanR'], y=df['epoch1_var'], ax=axes[1, 0])
# axes[1, 0].set_title('Epoch 0 Variance vs. Epoch 1 Reward')

# sns.scatterplot(x=df['epoch1_meanR'], y=df['epoch0_var'], ax=axes[1, 1])
# axes[1, 1].set_title('Epoch 1 Variance vs. Epoch 0 Reward')

# plt.tight_layout()

# # Save the figure
# plt.savefig('/homes/gws/yifangc/verl/scripts/variance_reward_relationships.png', dpi=300, bbox_inches='tight')
# plt.close(fig)

import pandas as pd
import numpy as np

dir_path = "/homes/gws/yifangc/verl/scripts/"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Load and prepare the data
# -------------------------------

# Replace 'your_file.csv' with the path to your updated CSV file
df = pd.read_csv(dir_path+"wandb_export_2025-03-06T14_20_47.185-08_00.csv")

# Replace "null" strings with NaN and convert all columns to numeric values
df.replace("null", np.nan, inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')
df.dropna(inplace=True)

available_epochs = []
# Let's assume you have epochs 0 to 3
for epoch in range(4):
    var_col = f"epoch{epoch}_var"
    mean_col = f"epoch{epoch}_meanR"
    if var_col in df.columns and mean_col in df.columns:
        available_epochs.append(epoch)

n_epochs = len(available_epochs)
print("Available epochs with both var and meanR:", available_epochs)

# -------------------------------
# 2. Determine available epochs and global axis limits
# -------------------------------
available_epochs = []
# Assuming epochs 0 to 3 are possible:
for epoch in range(4):
    var_col = f"epoch{epoch}_var"
    mean_col = f"epoch{epoch}_meanR"
    if var_col in df.columns and mean_col in df.columns:
        available_epochs.append(epoch)

n_epochs = len(available_epochs)
print("Available epochs with both var and meanR:", available_epochs)



# -------------------------------
# 3. Compute the global maximum bin count across epochs
# -------------------------------
# We'll use the same gridsize and extent for all epochs so that the bins are comparable.
gridsize = 30
global_max_count = 0
total_points = len(df)

for epoch in available_epochs:
    x = df[f"epoch{epoch}_var"]
    y = df[f"epoch{epoch}_meanR"]
    # Create a dummy axis to compute the hexbin counts
    fig_dummy, ax_dummy = plt.subplots()
    hb_dummy = ax_dummy.hexbin(x, y, gridsize=gridsize, cmap='viridis', mincnt=1,)
    counts = hb_dummy.get_array()
    if counts.size > 0:
        local_max = counts.max()
        if local_max > global_max_count:
            global_max_count = local_max
    plt.close(fig_dummy)

# Compute the global maximum percentage for labeling (each count divided by total_points)
global_max_percentage = (global_max_count / total_points) * 100
print(f"Global max count: {global_max_count}, corresponding to {global_max_percentage:.2f}%")

# -------------------------------
# 4. Create subplots with hexbin plots using a common color scale
# -------------------------------
fig, axes = plt.subplots(1, n_epochs, figsize=(6 * n_epochs, 6), squeeze=False)

for idx, epoch in enumerate(available_epochs):
    var_col = f"epoch{epoch}_var"
    mean_col = f"epoch{epoch}_meanR"
    ax = axes[0, idx]
    
    hb = ax.hexbin(df[var_col], df[mean_col],
                   gridsize=gridsize,
                   cmap='viridis',
                   mincnt=1,
                   vmin=0,
                   vmax=global_max_count)  # Fixed scale across subplots
    ax.set_xlabel(var_col)
    ax.set_ylabel(mean_col)
    ax.set_title(f"Epoch {epoch}: {var_col} vs {mean_col}")

    # Add a colorbar for each subplot and convert tick labels to percentage of total points
    cb = fig.colorbar(hb, ax=ax)
    ticks = cb.get_ticks()
    percent_ticks = [f"{(t / total_points * 100):.1f}%" for t in ticks]
    cb.set_ticks(ticks)
    cb.set_ticklabels(percent_ticks)
    cb.set_label("Percentage (%)")

plt.tight_layout()

# Save the combined figure to a file
combined_plot_path = dir_path+"combined_epochs_hexbin_percentage.png"
plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
plt.close()
