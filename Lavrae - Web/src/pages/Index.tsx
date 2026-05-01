import Layout from "@/components/layout/Layout";
import Hero from "@/components/Hero";
import CategoryGrid from "@/components/CategoryGrid";
import ReelSlider from "@/components/ReelSlider";
import ProductCard from "@/components/ProductCard";
import Marquee from "@/components/Marquee";
import { products } from "@/data/products";
import { Link } from "react-router-dom";

const Index = () => {
  const bestSellers = products.filter(p => p.tags.includes("Best Seller")).slice(0, 4);
  const featured = products.slice(0, 4);

  return (
    <Layout>
      <Hero />
      <Marquee />
      <CategoryGrid />

      {/* Best Sellers */}
      <section className="container py-20 md:py-28">
        <div className="flex items-end justify-between mb-10">
          <div>
            <p className="text-xs uppercase tracking-luxury text-accent mb-2">Most Loved</p>
            <h2 className="font-serif text-3xl md:text-5xl">Best sellers</h2>
          </div>
          <Link to="/shop" className="hidden md:inline-block text-xs uppercase tracking-luxury story-link">View all</Link>
        </div>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 md:gap-8">
          {bestSellers.map((p, i) => <ProductCard key={p.id} product={p} index={i} />)}
        </div>
      </section>

      <ReelSlider />

      {/* Featured */}
      <section className="bg-secondary/40 py-20 md:py-28">
        <div className="container">
          <div className="text-center max-w-xl mx-auto mb-12">
            <p className="text-xs uppercase tracking-luxury text-accent mb-2">Just In</p>
            <h2 className="font-serif text-3xl md:text-5xl mb-4">The new arrivals</h2>
            <p className="text-muted-foreground text-sm">Considered pieces, released in small runs.</p>
          </div>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 md:gap-8">
            {featured.map((p, i) => <ProductCard key={p.id} product={p} index={i} />)}
          </div>
          <div className="text-center mt-12">
            <Link
              to="/new-arrivals"
              className="inline-flex items-center justify-center px-8 py-3.5 border border-foreground text-foreground text-xs uppercase tracking-luxury hover:bg-foreground hover:text-background transition-colors duration-500"
            >
              View all arrivals
            </Link>
          </div>
        </div>
      </section>

      {/* Atelier */}
      <section className="container py-24 md:py-32 grid md:grid-cols-2 gap-12 md:gap-20 items-center">
        <div className="aspect-[4/5] bg-secondary hover-zoom rounded-sm">
          <img src={products[5].images[0]} alt="Atelier" loading="lazy" className="w-full h-full object-cover" />
        </div>
        <div>
          <p className="text-xs uppercase tracking-luxury text-accent mb-3">The Atelier</p>
          <h2 className="font-serif text-4xl md:text-6xl leading-tight mb-6">
            Tailored, never <em className="not-italic text-accent">manufactured.</em>
          </h2>
          <p className="text-muted-foreground leading-relaxed mb-4">
            Every V·tailers piece is shaped by hand in our small Mumbai studio. We work in
            limited capsules, choose fabrics that age beautifully, and finish each garment
            the way our grandmothers' tailors once did — with patience.
          </p>
          <p className="text-muted-foreground leading-relaxed mb-8">
            The result is a wardrobe that feels personal, weightless, and quietly royal.
          </p>
          <Link to="/shop" className="text-xs uppercase tracking-luxury story-link text-accent">
            Discover the craft
          </Link>
        </div>
      </section>
    </Layout>
  );
};

export default Index;
