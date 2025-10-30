"""
Streamlit Web Interface - IMPROVED

Clean, functional UI for the perturbation simulator.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

from simulate import PerturbationSimulator
from ingest import create_demo_dataset, load_h5ad


# ============================================================================
# PAGE CONFIG & STATE INITIALIZATION
# ============================================================================

st.set_page_config(
    page_title="SC Perturbation Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def init_simulator():
    """Initialize simulator once."""
    return PerturbationSimulator(verbose=False)

def get_simulator():
    """Get or create simulator."""
    if 'sim' not in st.session_state:
        st.session_state.sim = init_simulator()
    return st.session_state.sim

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_trajectory_3d(trajectory: np.ndarray):
    """Plot 3D trajectory in PCA space."""
    if trajectory.shape[2] < 3:
        st.warning("Need at least 3 dimensions for 3D plot")
        return
    
    traj_3d = trajectory[:, :, :3]
    fig = go.Figure()
    
    colors = px.colors.qualitative.Plotly
    for cell_idx in range(min(traj_3d.shape[1], 5)):
        x = traj_3d[:, cell_idx, 0]
        y = traj_3d[:, cell_idx, 1]
        z = traj_3d[:, cell_idx, 2]
        
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines+markers',
            name=f'Cell {cell_idx}',
            line=dict(color=colors[cell_idx % len(colors)], width=4),
            marker=dict(size=5)
        ))
    
    fig.update_layout(
        title='Simulated Trajectory (PCA Space)',
        scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'),
        height=500,
        hovermode='closest'
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_trajectory_comparison(wt_traj: np.ndarray, pert_traj: np.ndarray):
    """Compare WT vs perturbed."""
    fig = go.Figure()
    
    time_steps = wt_traj.shape[0]
    t = np.arange(time_steps)
    
    for cell_idx in range(min(wt_traj.shape[1], 3)):
        wt_val = wt_traj[:, cell_idx, 0]
        pert_val = pert_traj[:, cell_idx, 0]
        
        fig.add_trace(go.Scatter(
            x=t, y=wt_val, mode='lines+markers',
            name=f'WT Cell {cell_idx}',
            line=dict(dash='solid', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=t, y=pert_val, mode='lines+markers',
            name=f'Perturbed Cell {cell_idx}',
            line=dict(dash='dash', width=3)
        ))
    
    fig.update_layout(
        title='Wild-Type vs Perturbed Trajectories',
        xaxis_title='Time Step',
        yaxis_title='PC1 Value',
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_effect_size(wt_traj: np.ndarray, pert_traj: np.ndarray):
    """Plot L2 distance between WT and perturbed over time."""
    # Compute L2 distance at each time step
    distances = np.linalg.norm(pert_traj - wt_traj, axis=(1, 2))
    
    time_steps = np.arange(len(distances))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_steps,
        y=distances,
        mode='lines+markers',
        fill='tozeroy',
        name='L2 Distance',
        line=dict(color='red', width=3),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title='Perturbation Effect Size Over Time',
        xaxis_title='Time Step',
        yaxis_title='L2 Distance (WT vs Perturbed)',
        height=400,
        hovermode='x'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    return distances


# ============================================================================
# MAIN UI
# ============================================================================

def main():
    st.title("🧬 Single-Cell Perturbation Simulator")
    st.markdown("**Dynamic ODE-based perturbation prediction for single-cell RNA-seq**")
    
    sim = get_simulator()
    
    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================
    
    st.header("Step 1️⃣ Load Data", divider="blue")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Source")
        data_source = st.radio(
            "Choose data source:",
            ["📊 Demo Dataset", "📁 Upload .h5ad File"],
            label_visibility="collapsed"
        )
    
    with col2:
        st.subheader("Data Info")
        if sim.adata is not None:
            st.success(f"✅ **Loaded:** {sim.adata.n_obs} cells × {sim.adata.n_vars} genes")
        else:
            st.info("No data loaded yet")
    
    if data_source == "📊 Demo Dataset":
        col1, col2, col3 = st.columns(3)
        with col1:
            n_cells = st.number_input("Cells:", 50, 1000, 300)
        with col2:
            n_genes = st.number_input("Genes:", 200, 3000, 1000)
        with col3:
            if st.button("🚀 Load Demo", use_container_width=True):
                with st.spinner("Creating demo dataset..."):
                    sim.adata = create_demo_dataset(n_cells, n_genes)
                    st.success(f"✅ Demo loaded: {n_cells} cells × {n_genes} genes")
                    st.rerun()
    
    else:
        uploaded_file = st.file_uploader("Upload .h5ad file", type=['h5ad'])
        if uploaded_file is not None:
            with st.spinner("Loading..."):
                temp_path = Path("temp.h5ad")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                sim.adata = load_h5ad(str(temp_path))
                st.success(f"✅ Loaded: {sim.adata.n_obs} cells × {sim.adata.n_vars} genes")
                st.rerun()
    
    if sim.adata is None:
        st.stop()
    
    # ========================================================================
    # STEP 2: PREPROCESSING
    # ========================================================================
    
    st.header("Step 2️⃣ Preprocessing", divider="blue")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_top_genes = st.number_input("HVGs:", 200, 5000, 2000)
    with col2:
        n_pcs = st.number_input("PCA dims:", 5, 100, 50)
    with col3:
        n_neighbors = st.number_input("kNN:", 5, 100, 20)
    with col4:
        if st.button("⚙️ Preprocess", use_container_width=True):
            with st.spinner("Running preprocessing..."):
                sim.preprocess(
                    n_top_genes=n_top_genes,
                    n_pcs=n_pcs,
                    n_neighbors=n_neighbors
                )
            st.success("✅ Preprocessing complete!")
            st.rerun()
    
    if 'X_pca' not in sim.adata.obsm:
        st.info("👈 Run preprocessing to continue")
        st.stop()
    
    # ========================================================================
    # STEP 3: VELOCITY & VECTOR FIELD
    # ========================================================================
    
    st.header("Step 3️⃣ Velocity & Vector Field", divider="blue")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Compute Velocity", use_container_width=True):
            with st.spinner("Computing velocity..."):
                sim.compute_velocity(mode='stochastic')
            st.success("✅ Velocity computed!")
            st.rerun()
    
    with col2:
        if st.button("🎯 Fit Vector Field", use_container_width=True):
            with st.spinner("Fitting vector field..."):
                sim.fit_vector_field(method='knn_rbf', n_neighbors=30)
            st.success("✅ Vector field fitted!")
            st.rerun()
    
    with col3:
        if st.button("🧠 Train Decoder", use_container_width=True):
            with st.spinner("Training decoder (this may take 1-2 min)..."):
                sim.train_decoder(n_epochs=30, batch_size=32)
            st.success("✅ Decoder trained!")
            st.rerun()
    
    status_col1, status_col2, status_col3 = st.columns(3)
    with status_col1:
        vel_done = 'velocity_pca' in sim.adata.obsm
        st.metric("Velocity", "✅ Done" if vel_done else "⏳ Pending")
    with status_col2:
        vf_done = sim.vector_field is not None
        st.metric("Vector Field", "✅ Done" if vf_done else "⏳ Pending")
    with status_col3:
        dec_done = sim.decoder is not None
        st.metric("Decoder", "✅ Done" if dec_done else "⏳ Pending")
    
    if not (vel_done and vf_done and dec_done):
        st.info("👈 Complete all steps above to run simulations")
        st.stop()
    
    # ========================================================================
    # STEP 4: SIMULATION
    # ========================================================================
    
    st.header("Step 4️⃣ Run Simulations", divider="blue")
    
    sim_type = st.radio(
        "Choose simulation type:",
        ["🔬 Trajectory Simulation", "⚗️ Perturbation Simulation"],
        horizontal=True
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        t_max = st.number_input("Time duration:", 1.0, 100.0, 10.0)
    with col2:
        n_steps = st.number_input("Time steps:", 10, 500, 100)
    with col3:
        n_cells = st.number_input("Number of cells:", 1, 10, 3)
    
    # -------- TRAJECTORY SIMULATION --------
    if sim_type == "🔬 Trajectory Simulation":
        if st.button("Run Trajectory Simulation", use_container_width=True, type="primary"):
            x0 = sim.adata.obsm['X_pca'][:n_cells]
            
            with st.spinner("Simulating..."):
                result = sim.simulate_trajectory(
                    x0,
                    t_max=t_max,
                    n_steps=n_steps,
                    return_genes=True
                )
            
            st.success("✅ Simulation complete!")
            
            # Display results
            st.subheader("Results")
            plot_trajectory_3d(result['trajectory_pca'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Cells simulated", n_cells)
            with col2:
                st.metric("Time points", n_steps)
    
    # -------- PERTURBATION SIMULATION --------
    else:
        col1, col2 = st.columns(2)
        with col1:
            gene_idx = st.selectbox(
                "Gene to perturb:",
                range(min(50, len(sim.adata.var_names))),
                format_func=lambda i: sim.adata.var_names[i]
            )
            gene_name = sim.adata.var_names[gene_idx]
        
        with col2:
            fold_change = st.number_input("Fold change:", 0.1, 10.0, 2.0, step=0.1)
        
        if st.button("Run Perturbation Simulation", use_container_width=True, type="primary"):
            x0 = sim.adata.obsm['X_pca'][:n_cells]
            
            with st.spinner(f"Simulating {gene_name} perturbation (fold change: {fold_change}x)..."):
                result = sim.simulate_perturbation(
                    x0,
                    perturbation_gene=gene_name,
                    fold_change=fold_change,
                    t_max=t_max,
                    n_steps=n_steps
                )
            
            st.success("✅ Perturbation simulation complete!")
            
            # Display results
            st.subheader("Results")
            
            tab1, tab2, tab3 = st.tabs(["📈 Trajectory Comparison", "📊 Effect Size", "📋 Summary"])
            
            with tab1:
                plot_trajectory_comparison(
                    result['wt_trajectory_pca'],
                    result['perturbed_trajectory_pca']
                )
            
            with tab2:
                distances = plot_effect_size(
                    result['wt_trajectory_pca'],
                    result['perturbed_trajectory_pca']
                )
            
            with tab3:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Gene", gene_name)
                with col2:
                    st.metric("Fold Change", f"{fold_change}x")
                with col3:
                    st.metric("Max Effect", f"{distances.max():.4f}")
                with col4:
                    st.metric("Mean Effect", f"{distances.mean():.4f}")
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using Streamlit | Single-Cell RNA-seq Perturbation Prediction"
    )


if __name__ == "__main__":
    main()
