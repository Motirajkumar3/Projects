import { useRef } from "react";
import { ChevronLeft, ChevronRight, Play } from "lucide-react";
import reel1 from "@/assets/reel1.jpg";
import reel2 from "@/assets/reel2.jpg";
import reel3 from "@/assets/reel3.jpg";
import reel4 from "@/assets/reel4.jpg";
import reel5 from "@/assets/reel5.jpg";

const reels = [
  { src: reel1, title: "The Drape" },
  { src: reel2, title: "In the Atelier" },
  { src: reel3, title: "Maroon Detail" },
  { src: reel4, title: "Folded Stories" },
  { src: reel5, title: "Boutique Walk" },
];

const ReelSlider = () => {
  const ref = useRef<HTMLDivElement>(null);
  const scroll = (dir: 1 | -1) => {
    ref.current?.scrollBy({ left: dir * 320, behavior: "smooth" });
  };

  return (
    <section className="py-20 md:py-28">
      <div className="container">
        <div className="flex items-end justify-between mb-8">
          <div>
            <p className="text-xs uppercase tracking-luxury text-accent mb-2">@vtailers</p>
            <h2 className="font-serif text-3xl md:text-5xl">Behind the seams</h2>
          </div>
          <div className="hidden md:flex gap-2">
            <button onClick={() => scroll(-1)} aria-label="Previous" className="p-2.5 border border-border hover:border-accent hover:text-accent transition-colors">
              <ChevronLeft className="h-4 w-4" />
            </button>
            <button onClick={() => scroll(1)} aria-label="Next" className="p-2.5 border border-border hover:border-accent hover:text-accent transition-colors">
              <ChevronRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>

      <div
        ref={ref}
        className="flex gap-4 md:gap-6 overflow-x-auto scrollbar-none snap-x snap-mandatory px-6 md:px-[max(1.5rem,calc((100vw-1400px)/2+1.5rem))]"
      >
        {reels.map((r, i) => (
          <div
            key={i}
            className="snap-start shrink-0 w-[260px] md:w-[300px] aspect-[9/16] relative group cursor-pointer hover-zoom rounded-sm"
          >
            <img
              src={r.src}
              alt={r.title}
              loading="lazy"
              width={576}
              height={992}
              className="absolute inset-0 w-full h-full object-cover"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-foreground/60 via-transparent to-transparent" />
            <div className="absolute inset-0 flex items-center justify-center opacity-90 group-hover:opacity-100 transition">
              <div className="h-12 w-12 rounded-full bg-background/80 backdrop-blur flex items-center justify-center">
                <Play className="h-4 w-4 text-foreground ml-0.5" fill="currentColor" />
              </div>
            </div>
            <p className="absolute bottom-4 left-4 text-background text-sm font-serif">{r.title}</p>
          </div>
        ))}
      </div>
    </section>
  );
};

export default ReelSlider;
