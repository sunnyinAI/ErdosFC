// Erdos Platform site — tiny progressive enhancements (no dependencies).

// Reveal sections on scroll.
const io = new IntersectionObserver(
  (entries) => {
    for (const e of entries) {
      if (e.isIntersecting) {
        e.target.classList.add("in");
        io.unobserve(e.target);
      }
    }
  },
  { threshold: 0.12 }
);
document.querySelectorAll(".reveal").forEach((el) => io.observe(el));

// Copy-to-clipboard on code blocks.
document.querySelectorAll(".copy").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const text = btn.getAttribute("data-copy") || "";
    try {
      await navigator.clipboard.writeText(text);
      const original = btn.textContent;
      btn.textContent = "copied ✓";
      setTimeout(() => (btn.textContent = original), 1400);
    } catch {
      btn.textContent = "copy failed";
    }
  });
});
