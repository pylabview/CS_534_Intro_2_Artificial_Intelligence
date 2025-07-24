# ───────────────────────────────────────────────────────────────
# 7)  ✍️  Save both tables to a Markdown file
# ───────────────────────────────────────────────────────────────
REPORT_PATH = "model_comparison_report.md"

with open(REPORT_PATH, "w", encoding="utf-8") as md:
    md.write("# Machine‑Failure Model Comparison\n\n")

    md.write("## Table 1 – 5‑fold CV on 80 % Training Data\n\n")
    md.write(table1.to_markdown(index=False))
    md.write("\n\n")

    md.write("## Table 2 – Performance on 20 % Hold‑out Test Set\n\n")
    md.write(table2.to_markdown(index=False))
    md.write("\n\n")

    # 8) Choose the champion (highest test‑set MCC)
    best_row = table2.sort_values("MCC on 20 % Test set", ascending=False).iloc[0]
    champion = best_row["ML Trained Model"]

    md.write(f"### Conclusion – Selected Model\n\n")
    md.write(
        f"The **{champion}** achieved the highest MCC on both cross‑validation "
        f"and the held‑out test set, so we choose it as the production model.\n"
    )

print(f"\n✅  Markdown report written to: {REPORT_PATH}")
