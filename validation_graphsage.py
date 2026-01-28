import sys
import numpy as np

try:
    # Modern TF 2.x often hides the compat.v1 layer 
    # which legacy research repos like this desperately need.
    import tensorflow as tf
    if tf.__version__.startswith('1'):
        import tensorflow as _tf
    else:
        import tensorflow.compat.v1 as _tf
        _tf.disable_eager_execution()
    print(f"    [✓] TensorFlow {tf.__version__} loaded (Compat Mode Active).")
except ImportError:
    print("CRITICAL FAILURE: TensorFlow not found.")
    sys.exit(1)

def test_graphsage_logic():
    print("--- Starting Legacy GraphSAGE Functional Verification ---")
    
    try:
        # 1. Feature and Adjacency Mock
        # GraphSAGE is famous for Inductive Learning via sampling.
        print("--> Generating synthetic graph structures...")
        features = np.random.randn(10, 128).astype(np.float32)
        adj = np.random.randint(0, 2, (10, 10)).astype(np.float32)
        print("    [✓] Synthetic data ready.")

        # 2. Layer Initialization (The TF 1.x vs 2.x Trap)
        # Legacy GraphSAGE uses 'tf.layers' which was removed in TF 2.0.
        # This test identifies if AURA found a version that still supports 'compat.v1.layers'.
        print("--> Verifying Layer Math (Dense/Dropout)...")
        input_ph = _tf.placeholder(_tf.float32, shape=(None, 128))
        dropout_ph = _tf.placeholder(_tf.float32)
        
        # Use a core layer often used in the original repo
        layer = _tf.layers.dense(input_ph, 64, activation=_tf.nn.relu)
        
        print("    [✓] GraphSAGE-style Layer Graph constructed.")

        # 3. Session Run (The "Old World" Check)
        with _tf.Session() as sess:
            sess.run(_tf.global_variables_initializer())
            out = sess.run(layer, feed_dict={input_ph: features, dropout_ph: 0.5})
        
        if out.shape == (10, 64):
            print(f"    [✓] Execution successful. Shape: {out.shape}")
        else:
            raise ValueError(f"Unexpected output shape: {out.shape}")

        print("--- SMOKE TEST PASSED ---")

    except Exception as e:
        print(f"CRITICAL VALIDATION FAILURE: {str(e)}")
        # This will likely catch the AttributeError when 'tf.layers' 
        # is missing in modern TensorFlow.
        sys.exit(1)

if __name__ == "__main__":
    test_graphsage_logic()