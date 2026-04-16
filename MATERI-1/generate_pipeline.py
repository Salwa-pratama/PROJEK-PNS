import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(10, 7))

# Define pipeline steps
steps = [
    "Data Ingestion\n(Kaggle Credit Card)",
    "EDA & Data Cleaning\n(Check Missing values & Duplicates)",
    "Advanced Preprocessing\n(RobustScaler, Stratified Split)",
    "Scientific Balancing\n(SMOTE on Train Data)",
    "Model Architecture\n(LinearSVC Training)",
    "Performance Audit & XAI\n(Metrics, SHAP, What-If Simulation)"
]

# Coordinates
x_center = 0.5
y_start = 0.9
y_step = 0.15

for i, step in enumerate(steps):
    y = y_start - i*y_step
    # Draw box
    box = mpatches.FancyBboxPatch((x_center-0.3, y-0.05), 0.6, 0.1, 
                                  boxstyle="round,pad=0.03", 
                                  ec="#1D3557", fc="#E8F1F2", lw=2)
    ax.add_patch(box)
    ax.text(x_center, y, step, ha="center", va="center", fontsize=11, fontweight='bold', color="#1D3557")
    
    # Draw arrow
    if i < len(steps)-1:
        ax.annotate('', xy=(x_center, y-0.05), xytext=(x_center, y-0.05-y_step+0.1),
                    arrowprops=dict(arrowstyle="->", lw=2, color="#E63946"))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_axis_off()
plt.title("Pipeline Arsitektur Pemodelan: Fraud Detection", fontsize=15, fontweight='bold', color="#333333", y=0.95)
plt.tight_layout()

# Save the diagram to MATERI-1 directory
output_path = '/media/pratama/Data/Other/SELURUH FILE PERKULIAHAN/MATERI SMESTER 6/Pemodelan dan simulasi/projek/KODING-KASUS/MATERI-1/Pipeline_Arsitektur.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Pipeline image successfully saved to {output_path}")
