import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle

# --- 1. CONFIGURATION ---

# Define the actual folder name on your hard drive
# This must match the folder where your images are located
IMAGE_DATASET_FOLDER = "seg_test"

st.set_page_config(layout="wide", page_title="Advanced Image Retrieval")

@st.cache_resource
def load_model():
    # CLIP model handles both Text and Image to Vector conversion
    return SentenceTransformer('clip-ViT-B-32')

model = load_model()

# --- 2. SIDEBAR: CONTROL PANEL ---
st.sidebar.title("Search Settings")

# Section A: Input
st.sidebar.subheader("1. Input Query")
input_type = st.sidebar.radio("Query Type", ["Text", "Image"])

# Section B: Advanced Parameters (Slide 13 & 15)
st.sidebar.markdown("---")
st.sidebar.subheader("2. Feedback Algorithms")

# Algorithm Selection
feedback_method = st.sidebar.selectbox(
    "Reformulation Method",
    ["Standard Rocchio", "Ide Dec Hi"]
)

# [cite_start]Tunable Weights [cite: 322-324]
with st.sidebar.expander("Adjust Weights", expanded=False):
    st.markdown("Control impact of feedback:")
    ALPHA = st.slider("Alpha (Original Query)", 0.0, 2.0, 1.0, 0.1, help="Keep original intent")
    BETA = st.slider("Beta (Relevant)", 0.0, 2.0, 0.75, 0.1, help="Move towards relevant")
    GAMMA = st.slider("Gamma (Non-Relevant)", 0.0, 2.0, 0.25, 0.1, help="Move away from irrelevant")

# Section C: View Settings
st.sidebar.markdown("---")
st.sidebar.subheader("3. View Settings")
TOP_K = st.sidebar.slider("Number of Results (k)", 1, 50, 10, 1)
GRID_COLS = st.sidebar.slider("Grid Columns", 2, 8, 5, 1)


# --- 3. DATASET LOADING ---
if 'db_vectors' not in st.session_state:
    try:
        # Load the pre-calculated vectors
        st.session_state.db_vectors = np.load('db_vectors.npy')
        
        # Load the filenames
        with open('image_filenames.pkl', 'rb') as f:
            st.session_state.image_names = pickle.load(f)
            
        st.success(f"Loaded {len(st.session_state.db_vectors)} images from database.")
        
    except FileNotFoundError:
        st.error("Dataset files not found! Please run 'prepare_data.py' first.")
        # Fallback for testing UI without data
        st.session_state.db_vectors = np.random.rand(10, 512)
        st.session_state.image_names = [f"demo_{i}.jpg" for i in range(10)]

# --- 4. ALGORITHMS ---

def retrieve_images(query_vector, db_vectors, top_k=5):
    # Calculate Cosine Similarity
    similarities = cosine_similarity([query_vector], db_vectors)[0]
    # [cite_start]Get indices of top k scores [cite: 295]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return top_indices

def rocchio_update(original_query, relevant_vecs, irrelevant_vecs):
    # Implementation of Slide 13: Standard Rocchio
    # [cite_start]Uses Mean (Average) of vectors [cite: 321]
    
    q_new = ALPHA * original_query
    
    if len(relevant_vecs) > 0:
        mean_relevant = np.mean(relevant_vecs, axis=0)
        q_new += BETA * mean_relevant
        
    if len(irrelevant_vecs) > 0:
        mean_irrelevant = np.mean(irrelevant_vecs, axis=0)
        q_new -= GAMMA * mean_irrelevant
        
    return q_new

def ide_dec_hi_update(original_query, relevant_vecs, irrelevant_vecs):
    # [cite_start]Implementation of Slide 15: Ide "Dec Hi" Method [cite: 338]
    # Uses Sum (not mean) for relevant, and only the MAX irrelevant
    
    q_new = ALPHA * original_query
    
    if len(relevant_vecs) > 0:
        # Ide method uses Summation, not Mean
        q_new += BETA * np.sum(relevant_vecs, axis=0)
        
    if len(irrelevant_vecs) > 0:
        # Ide Dec Hi uses only the "highest ranked" (most interfering) negative image.
        # Since retrieve_images returns sorted results, the first one in this list 
        # (which comes from the top results) is likely the highest ranked one.
        most_interfering_vector = irrelevant_vecs[0]
        q_new -= GAMMA * most_interfering_vector
        
    return q_new

# --- 5. MAIN UI LOGIC ---
st.title("Interactive Image Retrieval System")

# Initialize Session State
if 'query_vector' not in st.session_state:
    st.session_state.query_vector = None
if 'iteration' not in st.session_state:
    st.session_state.iteration = 0

# --- STEP A: INPUT HANDLING ---
if input_type == "Text":
    text_query = st.sidebar.text_input("Enter description:")
    if st.sidebar.button("Search Text"):
        st.session_state.query_vector = model.encode(text_query)
        st.session_state.iteration = 1
        st.rerun()

elif input_type == "Image":
    img_file = st.sidebar.file_uploader("Upload Image", type=['jpg', 'png'])
    if img_file and st.sidebar.button("Search Image"):
        image = Image.open(img_file)
        st.session_state.query_vector = model.encode(image)
        st.session_state.iteration = 1
        st.rerun()

# --- STEP B: DISPLAY RESULTS ---
if st.session_state.query_vector is not None:
    st.subheader(f"Results (Iteration {st.session_state.iteration})")
    
    # Perform Retrieval using dynamic TOP_K
    top_indices = retrieve_images(st.session_state.query_vector, st.session_state.db_vectors, top_k=TOP_K)
    
    # Feedback Form
    with st.form("feedback_form"):
        # Use dynamic Grid Columns
        cols = st.columns(GRID_COLS)
        
        selected_relevant = []
        selected_irrelevant = []
        
        for idx, db_idx in enumerate(top_indices):
            # Wrap images across columns
            col = cols[idx % GRID_COLS]
            
            # --- ROBUST IMAGE DISPLAY ---
            stored_filename = st.session_state.image_names[db_idx]
            # Combine root folder + stored relative path
            full_image_path = os.path.join(IMAGE_DATASET_FOLDER, stored_filename)

            try:
                image = Image.open(full_image_path)
                # Correct parameter for recent Streamlit versions
                col.image(image, width="stretch") 
            except Exception as e:
                col.error("Image not found")
                # Fallback placeholder
                col.image("https://placehold.co/200x200/png?text=Missing", width="stretch")

            # --- FEEDBACK CHECKBOXES ---
            # Unique keys for every checkbox to avoid collisions
            is_rel = col.checkbox(f"Relevant", key=f"rel_{st.session_state.iteration}_{idx}")
            is_irrel = col.checkbox(f"Not Relevant", key=f"irr_{st.session_state.iteration}_{idx}")
            
            if is_rel:
                selected_relevant.append(st.session_state.db_vectors[db_idx])
            if is_irrel:
                selected_irrelevant.append(st.session_state.db_vectors[db_idx])

        st.markdown("---")
        
        # --- STEP C: REFINEMENT INPUTS ---
        col1, col2 = st.columns([3, 1])
        with col1:
             # Feature 3: Text Modification (Slide 20)
            text_modifier = st.text_input(
                "💬 Modify query with text (optional):", 
                placeholder="e.g. 'make it darker', 'remove trees'"
            )
        with col2:
            st.write("") # Spacer
            st.write("") 
            submit_feedback = st.form_submit_button("Refine Search (Apply Feedback)")
        
        # --- PROCESS FEEDBACK ---
        if submit_feedback:
            # 1. Apply Text Modifier (Vector Arithmetic)
            if text_modifier:
                text_vec = model.encode(text_modifier)
                # Weighted addition of text concept (Slide 20 logic)
                st.session_state.query_vector += (0.5 * text_vec)
                st.success(f"Applied text modifier: '{text_modifier}'")

            # 2. Apply Relevance Feedback
            if selected_relevant or selected_irrelevant:
                rel_array = np.array(selected_relevant)
                irr_array = np.array(selected_irrelevant)
                
                # Select Algorithm based on Sidebar
                if feedback_method == "Ide Dec Hi":
                    new_query = ide_dec_hi_update(
                        st.session_state.query_vector,
                        rel_array,
                        irr_array
                    )
                else:
                    new_query = rocchio_update(
                        st.session_state.query_vector,
                        rel_array,
                        irr_array
                    )
                
                st.session_state.query_vector = new_query
                st.session_state.iteration += 1
                st.rerun()
            elif not text_modifier:
                st.warning("Please mark images OR enter text to refine.")

if st.session_state.query_vector is None:
    st.info("👈 Upload an image or enter text in the sidebar to start.")