try:
    import hailo_dataflow_compiler
    print("✅ Hailo Dataflow Compiler is installed.")
    print("Version:", hailo_dataflow_compiler.__version__)
except ImportError as e:
    print("❌ Hailo Dataflow Compiler not found.")
    print("Error:", e)
