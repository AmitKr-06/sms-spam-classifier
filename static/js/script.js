// Placeholder for future enhancements (e.g. client-side validation,
// disabling the submit button while the form is processing, etc.)
document.addEventListener("DOMContentLoaded", () => {
    const form = document.querySelector("form");
    if (form) {
        form.addEventListener("submit", () => {
            const btn = form.querySelector("button");
            if (btn) {
                btn.disabled = true;
                btn.textContent = "Classifying...";
            }
        });
    }
});