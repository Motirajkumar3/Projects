const items = [
  "Hand-stitched in India",
  "Made-to-measure tailoring",
  "Limited capsule editions",
  "Complimentary alterations",
];

const Marquee = () => (
  <div className="bg-foreground text-background py-3 overflow-hidden border-y border-foreground">
    <div className="flex animate-marquee whitespace-nowrap">
      {[...items, ...items, ...items, ...items].map((t, i) => (
        <span key={i} className="mx-10 text-[11px] uppercase tracking-luxury inline-flex items-center gap-10">
          {t}
          <span className="text-accent">◆</span>
        </span>
      ))}
    </div>
  </div>
);

export default Marquee;
