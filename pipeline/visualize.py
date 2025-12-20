
import streamlit as st
import json
import pandas as pd
from pathlib import Path
import sys

# Add current dir to path to import local modules
sys.path.append(str(Path(__file__).parent))

from data_loader import load_val_data, load_test_data, Question

st.set_page_config(layout="wide", page_title="VNPT AI Results Viewer")

@st.cache_data
def load_data(dataset_name):
    if dataset_name == "val":
        return load_val_data()
    else:
        return load_test_data()

@st.cache_data
def load_results(model_name):
    path = Path(f"outputs/results_{model_name}_.json")
    if not path.exists():
        return None
    with open(path, 'r') as f:
        return json.load(f)

def main():
    st.title("VNPT AI Pipeline Results Viewer")

    # Sidebar
    st.sidebar.header("Configuration")
    dataset = st.sidebar.selectbox("Dataset", ["val", "test"])
    model = st.sidebar.selectbox("Model Results", ["small", "large"])
    
    # Load Data
    questions = load_data(dataset)
    results = load_results(model)
    
    if not results:
        st.error(f"No results found for model '{model}'. Run the pipeline first.")
        return

    # Create Dictionary for quick lookup
    q_dict = {q.qid: q for q in questions}
    r_dict = {r['qid']: r for r in results}
    
    # Merge Data
    data = []
    for q in questions:
        res = r_dict.get(q.qid, {})
        predicted = res.get('predicted', 'N/A')
        ground_truth = res.get('ground_truth', 'N/A')
        
        # Determine correctness
        is_correct = False
        if q.answer:
            is_correct = predicted == q.answer
        
        # Prepare row
        row = {
            "QID": q.qid,
            "Question": q.raw_question[:100] + "..." if q.raw_question else q.question[:100] + "...",
            "Predicted": predicted,
            "Ground Truth": ground_truth,
            "Correct": is_correct,
            "Has Context": q.has_context(),
            "Full Question": q
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Metrics
    processed_count = len(results)
    total_count = len(questions)
    
    # Filter only those with ground truth for accuracy
    with_gt = df[df["Ground Truth"] != "N/A"]
    correct_count = with_gt[with_gt["Correct"] == True].shape[0]
    total_gt = with_gt.shape[0]
    accuracy = (correct_count / total_gt * 100) if total_gt > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Questions Processed", f"{processed_count}/{total_count}")
    col2.metric("Accuracy", f"{accuracy:.2f}%")
    col3.metric("Errors", total_gt - correct_count)
    
    # Filters
    st.subheader("Results Inspector")
    show_errors_only = st.checkbox("Show Errors Only")
    
    if show_errors_only:
        # Show incorrectly answered questions (and excludes those without ground truth if any)
        # But we mostly care about where Ground Truth exists and Prediction != Ground Truth
        filtered_df = with_gt[with_gt["Correct"] == False]
    else:
        filtered_df = df
        
    # Display Table
    # Use dataframe with column config for better display
    st.dataframe(
        filtered_df[["QID", "Question", "Predicted", "Ground Truth", "Correct"]],
        use_container_width=True,
        column_config={
            "Correct": st.column_config.CheckboxColumn(
                "Correct",
                help="Is the prediction correct?",
                disabled=True,
            ),
        },
        selection_mode="single-row",
        on_select="rerun",
        key="table_selection" 
    )
    
    # Detail View
    # Streamlit 1.35+ supports on_select. If older, we might need a selectbox.
    # Let's add a selectbox as fallback/primary navigation for details
    
    st.markdown("---")
    st.subheader("Detail View")
    
    # Get list of QIDs from filtered view
    qid_options = filtered_df["QID"].tolist()
    
    if not qid_options:
        st.info("No questions to display.")
        return
        
    selected_qid = st.selectbox("Select Question to Inspect", qid_options)
    
    if selected_qid:
        row = df[df["QID"] == selected_qid].iloc[0]
        q : Question = row["Full Question"]
        
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.markdown(f"**Question:** {q.raw_question if q.raw_question else q.question}")
            
            if q.has_context():
                with st.expander("Context", expanded=True):
                    st.markdown(q.context)
            
            st.markdown("**Choices:**")
            import string
            letters = string.ascii_uppercase
            for i, choice in enumerate(q.choices):
                if i < len(letters):
                    st.write(f"- **{letters[i]}**: {choice}")
        
        with c2:
            st.markdown("### Result")
            if row["Correct"]:
                st.success(f"**Predicted:** {row['Predicted']}")
            else:
                st.error(f"**Predicted:** {row['Predicted']}")
                st.info(f"**Ground Truth:** {row['Ground Truth']}")
            
            st.markdown(f"**Has Context:** {q.has_context()}")

if __name__ == "__main__":
    main()
