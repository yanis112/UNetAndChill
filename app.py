import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image

# Page setup
st.header("Image Coordinate Extractor")

# Sidebar controls
img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg'])
box_color = st.sidebar.color_picker(label="Box Color", value='#0000FF')
aspect_choice = st.sidebar.radio(label="Aspect Ratio", 
    options=["1:1", "16:9", "4:3", "2:3", "Free"])

aspect_dict = {
    "1:1": (1, 1),
    "16:9": (16, 9),
    "4:3": (4, 3),
    "2:3": (2, 3),
    "Free": None
}
aspect_ratio = aspect_dict[aspect_choice]

if img_file:
    img = Image.open(img_file)
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        cropped_box = st_cropper(
            img, 
            realtime_update=True, 
            box_color=box_color,
            aspect_ratio=aspect_ratio,
            return_type='box'  # Returns box coordinates
        )
    
    with col2:
        st.subheader("Crop Coordinates")
        if isinstance(cropped_box, dict):
            # Access coordinates from dictionary
            left = cropped_box.get('left', 0)
            top = cropped_box.get('top', 0)
            width = cropped_box.get('width', 0)
            height = cropped_box.get('height', 0)
            
            # Display coordinates
            st.write("Left:", round(left, 2))
            st.write("Top:", round(top, 2))
            st.write("Width:", round(width, 2))
            st.write("Height:", round(height, 2))
            
            # Calculate and display corner points
            st.write("Corner Points:")
            st.write(f"Top Left: ({round(left, 2)}, {round(top, 2)})")
            st.write(f"Top Right: ({round(left + width, 2)}, {round(top, 2)})")
            st.write(f"Bottom Left: ({round(left, 2)}, {round(top + height, 2)})")
            st.write(f"Bottom Right: ({round(left + width, 2)}, {round(top + height, 2)})")