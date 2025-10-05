from ui import CouponDetectorApp

app = CouponDetectorApp()
selected_sku = app.run()
print(f"Selected SKU: {selected_sku}")