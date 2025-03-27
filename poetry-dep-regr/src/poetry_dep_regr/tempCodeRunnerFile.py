# --- ГИСТОГРАММА ОСТАТКОВ ---
fig2, ax2 = plt.subplots()
residuals = bst.y - bst.y_pred
ax2.hist(residuals, bins='auto', density=True, color='blue', alpha=0.7)
ax2.set_title("Гистограмма остатков")
ax2.set_xlabel("Остатки")
ax2.set_ylabel("Частота")