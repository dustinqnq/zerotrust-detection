import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# Set English font
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Create figure with fixed size
fig, ax = plt.subplots(1, 1, figsize=(18, 12))
ax.set_xlim(0, 18)
ax.set_ylim(0, 12)
ax.axis('off')

# Define colors
color_stage1 = '#E3F2FD'  # Light blue
color_stage2 = '#FFF3E0'  # Light orange
color_stage3 = '#E8F5E8'  # Light green
color_decision = '#FCE4EC'  # Light pink
color_data = '#F3E5F5'    # Light purple

# Draw data input (centered at top)
data_box = FancyBboxPatch((7, 10.2), 4, 1, boxstyle="round,pad=0.1", 
                         facecolor=color_data, edgecolor='black', linewidth=2)
ax.add_patch(data_box)
ax.text(9, 10.7, 'Network Traffic Data\n& Feature Engineering', ha='center', va='center', fontsize=11, weight='bold')

# Define stage positions for perfect alignment
stage1_x, stage1_y = 1, 7
stage2_x, stage2_y = 6.75, 7  
stage3_x, stage3_y = 12.5, 7
stage_width, stage_height = 4.5, 2.5

# Draw Stage 1 (left)
stage1_box = FancyBboxPatch((stage1_x, stage1_y), stage_width, stage_height, boxstyle="round,pad=0.1", 
                           facecolor=color_stage1, edgecolor='blue', linewidth=2)
ax.add_patch(stage1_box)
ax.text(stage1_x + stage_width/2, stage1_y + stage_height - 0.4, 'Stage 1: Fast Detection', 
        ha='center', va='center', fontsize=12, weight='bold')

# Stage 1 internal components
shallow_binary = FancyBboxPatch((stage1_x + 0.4, stage1_y + 0.8), 1.6, 0.6, boxstyle="round,pad=0.05", 
                               facecolor='white', edgecolor='blue', linewidth=1.5)
ax.add_patch(shallow_binary)
ax.text(stage1_x + 1.2, stage1_y + 1.1, 'Shallow Binary\nClassifier', ha='center', va='center', fontsize=9)

shallow_multi = FancyBboxPatch((stage1_x + 2.5, stage1_y + 0.8), 1.6, 0.6, boxstyle="round,pad=0.05", 
                              facecolor='white', edgecolor='blue', linewidth=1.5)
ax.add_patch(shallow_multi)
ax.text(stage1_x + 3.3, stage1_y + 1.1, 'Shallow Multi-class\nClassifier', ha='center', va='center', fontsize=9)

# Draw Stage 2 (center)
stage2_box = FancyBboxPatch((stage2_x, stage2_y), stage_width, stage_height, boxstyle="round,pad=0.1", 
                           facecolor=color_stage2, edgecolor='orange', linewidth=2)
ax.add_patch(stage2_box)
ax.text(stage2_x + stage_width/2, stage2_y + stage_height - 0.4, 'Stage 2: Deep Detection', 
        ha='center', va='center', fontsize=12, weight='bold')

# Stage 2 internal components
deep_binary = FancyBboxPatch((stage2_x + 0.4, stage2_y + 0.8), 1.6, 0.6, boxstyle="round,pad=0.05", 
                            facecolor='white', edgecolor='orange', linewidth=1.5)
ax.add_patch(deep_binary)
ax.text(stage2_x + 1.2, stage2_y + 1.1, 'Deep Binary\nClassifier', ha='center', va='center', fontsize=9)

deep_multi = FancyBboxPatch((stage2_x + 2.5, stage2_y + 0.8), 1.6, 0.6, boxstyle="round,pad=0.05", 
                           facecolor='white', edgecolor='orange', linewidth=1.5)
ax.add_patch(deep_multi)
ax.text(stage2_x + 3.3, stage2_y + 1.1, 'Deep Multi-class\nClassifier', ha='center', va='center', fontsize=9)

# Draw Stage 3 (right)
stage3_box = FancyBboxPatch((stage3_x, stage3_y), stage_width, stage_height, boxstyle="round,pad=0.1", 
                           facecolor=color_stage3, edgecolor='green', linewidth=2)
ax.add_patch(stage3_box)
ax.text(stage3_x + stage_width/2, stage3_y + stage_height - 0.4, 'Stage 3: Unknown\nAttack Detection', 
        ha='center', va='center', fontsize=12, weight='bold')

# Stage 3 internal components
autoencoder = FancyBboxPatch((stage3_x + 0.4, stage3_y + 0.8), 3.7, 0.6, boxstyle="round,pad=0.05", 
                            facecolor='white', edgecolor='green', linewidth=1.5)
ax.add_patch(autoencoder)
ax.text(stage3_x + 2.25, stage3_y + 1.1, 'Autoencoder +\nAnomaly Detection', ha='center', va='center', fontsize=9)

# Draw decision outputs in organized layout
decision_y = 4.5
decision_width, decision_height = 1.8, 1
decision_boxes = [
    # Stage 1 outputs
    {'pos': (stage1_x + 0.6, decision_y), 'text': 'Benign Traffic\nStage 1 Pass', 'color': '#4CAF50'},
    {'pos': (stage1_x + 2.8, decision_y), 'text': 'Known Attack\nStage 1 Alert', 'color': '#FF9800'},
    
    # Stage 2 outputs
    {'pos': (stage2_x + 0.6, decision_y), 'text': 'Benign Traffic\nStage 2 Pass', 'color': '#4CAF50'},
    {'pos': (stage2_x + 2.8, decision_y), 'text': 'Known Attack\nStage 2 Alert', 'color': '#FF5722'},
    
    # Stage 3 outputs
    {'pos': (stage3_x + 0.6, decision_y), 'text': 'Benign Traffic\nStage 3 Pass', 'color': '#4CAF50'},
    {'pos': (stage3_x + 2.8, decision_y), 'text': 'Unknown Attack\nStage 3 Alert', 'color': '#F44336'}
]

for box in decision_boxes:
    decision_box = FancyBboxPatch(box['pos'], decision_width, decision_height, boxstyle="round,pad=0.05", 
                                 facecolor=color_decision, edgecolor=box['color'], linewidth=2)
    ax.add_patch(decision_box)
    ax.text(box['pos'][0] + decision_width/2, box['pos'][1] + decision_height/2, box['text'], 
           ha='center', va='center', fontsize=9, weight='bold')

# Calculate precise arrow positions to align with box edges
# Data input to Stage 1 (from bottom of data box to top of stage1 box)
data_bottom = 10.2
stage1_top = stage1_y + stage_height
ax.arrow(9, data_bottom, -5.75, -(data_bottom - stage1_top - 0.1), 
         head_width=0.12, head_length=0.15, fc='black', ec='black', linewidth=2)

# Stage internal connections (aligned to component centers)
# Stage 1: binary to multi-class
ax.arrow(stage1_x + 2, stage1_y + 1.1, 0.4, 0, 
         head_width=0.08, head_length=0.12, fc='blue', ec='blue', linewidth=1.5)

# Stage 2: binary to multi-class  
ax.arrow(stage2_x + 2, stage2_y + 1.1, 0.4, 0, 
         head_width=0.08, head_length=0.12, fc='orange', ec='orange', linewidth=1.5)

# Stage outputs to decisions (from bottom of components to top of decision boxes)
component_bottom = stage1_y + 0.8
decision_top = decision_y + decision_height

# Stage 1 outputs
ax.arrow(stage1_x + 1.2, component_bottom, 0, -(component_bottom - decision_top - 0.1), 
         head_width=0.1, head_length=0.15, fc='green', ec='green', linewidth=2)
ax.arrow(stage1_x + 3.3, component_bottom, 0, -(component_bottom - decision_top - 0.1), 
         head_width=0.1, head_length=0.15, fc='orange', ec='orange', linewidth=2)

# Stage 2 outputs  
ax.arrow(stage2_x + 1.2, component_bottom, 0, -(component_bottom - decision_top - 0.1), 
         head_width=0.1, head_length=0.15, fc='green', ec='green', linewidth=2)
ax.arrow(stage2_x + 3.3, component_bottom, 0, -(component_bottom - decision_top - 0.1), 
         head_width=0.1, head_length=0.15, fc='orange', ec='orange', linewidth=2)

# Stage 3 outputs
ax.arrow(stage3_x + 1.5, component_bottom, 0, -(component_bottom - decision_top - 0.1), 
         head_width=0.1, head_length=0.15, fc='green', ec='green', linewidth=2)
ax.arrow(stage3_x + 3.5, component_bottom, 0, -(component_bottom - decision_top - 0.1), 
         head_width=0.1, head_length=0.15, fc='red', ec='red', linewidth=2)

# Inter-stage connections (from right edge of stage to left edge of next stage)
stage_center_y = stage1_y + stage_height/2

# Stage 1 to Stage 2
ax.arrow(stage1_x + stage_width, stage_center_y, 
         stage2_x - (stage1_x + stage_width) - 0.1, 0, 
         head_width=0.12, head_length=0.15, fc='red', ec='red', linewidth=2)
ax.text((stage1_x + stage_width + stage2_x)/2, stage_center_y + 0.3, 'Uncertain', 
        ha='center', va='center', fontsize=10, color='red', weight='bold')

# Stage 2 to Stage 3
ax.arrow(stage2_x + stage_width, stage_center_y, 
         stage3_x - (stage2_x + stage_width) - 0.1, 0, 
         head_width=0.12, head_length=0.15, fc='red', ec='red', linewidth=2)
ax.text((stage2_x + stage_width + stage3_x)/2, stage_center_y + 0.3, 'Unknown\nPattern', 
        ha='center', va='center', fontsize=10, color='red', weight='bold')

# Add labels for decision arrows
decision_labels = [
    {'pos': (stage1_x + 1.2, 6), 'text': 'Benign', 'color': 'green'},
    {'pos': (stage1_x + 3.3, 6), 'text': 'Known\nAttack', 'color': 'orange'},
    {'pos': (stage2_x + 1.2, 6), 'text': 'Benign', 'color': 'green'},
    {'pos': (stage2_x + 3.3, 6), 'text': 'Known\nAttack', 'color': 'orange'},
    {'pos': (stage3_x + 1.5, 6), 'text': 'Normal', 'color': 'green'},
    {'pos': (stage3_x + 3.5, 6), 'text': 'Anomaly', 'color': 'red'}
]

for label in decision_labels:
    ax.text(label['pos'][0], label['pos'][1], label['text'], 
           ha='center', va='center', fontsize=9, color=label['color'], weight='bold')

# Add title
ax.text(9, 11.5, 'Multi-Stage Zero Trust Intrusion Detection System Architecture', 
       ha='center', va='center', fontsize=16, weight='bold')

# Add legend
legend_elements = [
    patches.Patch(color=color_stage1, label='Fast Detection Stage'),
    patches.Patch(color=color_stage2, label='Deep Detection Stage'),
    patches.Patch(color=color_stage3, label='Unknown Detection Stage'),
    patches.Patch(color=color_decision, label='Detection Results')
]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.95), fontsize=10)

plt.tight_layout()
plt.savefig('zero_trust_ids_flowchart.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("Optimized flowchart saved as 'zero_trust_ids_flowchart.png'") 