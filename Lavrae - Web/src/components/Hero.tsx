import { Link } from "react-router-dom";
import hero from "@/assets/hero.jpg";

const Hero = () => (
  <section className="relative h-[92vh] min-h-[600px] w-full overflow-hidden">
    <img
      src={hero}
      alt="V-tailers signature piece"
      width={1600}
      height={1280}
      className="absolute inset-0 w-full h-full object-cover object-center animate-scale-in"
    />
    <div className="absolute inset-0 bg-gradient-hero" />
    <div className="absolute inset-0 bg-foreground/5" />

    <div className="relative h-full container flex flex-col justify-end pb-20 md:pb-28">
      <div className="max-w-xl animate-fade-up">
        <p className="text-xs uppercase tracking-luxury text-accent mb-4">Spring Edit · 2026</p>
        <h1 className="font-serif text-5xl md:text-7xl lg:text-8xl leading-[0.95] text-balance">
          Quietly <em className="not-italic text-accent">royal.</em><br />
          Tailored to you.
        </h1>
        <p className="mt-6 text-base md:text-lg text-muted-foreground max-w-md leading-relaxed">
          Heritage craft, modern restraint. A wardrobe that whispers, never shouts.
        </p>
        <div className="mt-10 flex flex-wrap gap-4">
          <Link
            to="/new-arrivals"
            className="inline-flex items-center justify-center px-7 py-3.5 bg-accent text-accent-foreground text-xs uppercase tracking-luxury hover:bg-foreground transition-colors duration-500"
          >
            Shop New Arrivals
          </Link>
          <Link
            to="/shop"
            className="inline-flex items-center justify-center px-7 py-3.5 border border-foreground/30 text-foreground text-xs uppercase tracking-luxury hover:border-foreground hover:bg-foreground hover:text-background transition-colors duration-500"
          >
            The Collection
          </Link>
        </div>
      </div>
    </div>
  </section>
);

export default Hero;
