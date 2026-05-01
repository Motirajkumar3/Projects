import { Link } from "react-router-dom";
import { categories } from "@/data/products";

const CategoryGrid = () => (
  <section className="py-20 md:py-28 container">
    <div className="text-center mb-14 max-w-xl mx-auto animate-fade-up">
      <p className="text-xs uppercase tracking-luxury text-accent mb-3">The Edit</p>
      <h2 className="font-serif text-3xl md:text-5xl mb-4">Shop by category</h2>
      <p className="text-muted-foreground text-sm leading-relaxed">
        Pieces tailored with intention — for the wardrobe you actually wear.
      </p>
    </div>

    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 md:gap-6">
      {categories.map((c, i) => (
        <Link
          key={c.slug}
          to={`/category/${c.slug}`}
          className="group relative aspect-[3/4] overflow-hidden hover-zoom rounded-sm animate-fade-up"
          style={{ animationDelay: `${i * 80}ms` }}
        >
          <img
            src={c.image}
            alt={c.name}
            loading="lazy"
            width={800}
            height={1000}
            className="absolute inset-0 w-full h-full object-cover"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-foreground/55 via-foreground/10 to-transparent" />
          <div className="absolute bottom-5 left-5 right-5 text-background">
            <p className="text-[10px] uppercase tracking-luxury opacity-80">{c.blurb}</p>
            <h3 className="font-serif text-2xl md:text-3xl mt-1">{c.name}</h3>
            <span className="story-link text-xs uppercase tracking-luxury inline-block mt-2">Explore</span>
          </div>
        </Link>
      ))}
    </div>
  </section>
);

export default CategoryGrid;
