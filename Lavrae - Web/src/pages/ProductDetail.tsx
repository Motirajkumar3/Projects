import { useParams, Link } from "react-router-dom";
import Layout from "@/components/layout/Layout";
import { products, formatINR } from "@/data/products";
import ProductCard from "@/components/ProductCard";
import { useState } from "react";
import { Heart, Truck, Scissors, RotateCcw } from "lucide-react";

const ProductDetail = () => {
  const { slug } = useParams();
  const product = products.find(p => p.slug === slug);
  const [active, setActive] = useState(0);

  if (!product) {
    return (
      <Layout>
        <div className="container pt-40 pb-20 text-center">
          <h1 className="font-serif text-4xl mb-4">Piece not found</h1>
          <Link to="/shop" className="text-accent story-link text-xs uppercase tracking-luxury">Back to shop</Link>
        </div>
      </Layout>
    );
  }

  const related = products.filter(p => p.category === product.category && p.id !== product.id).slice(0, 4);

  return (
    <Layout>
      <section className="pt-28 md:pt-32 container">
        <p className="text-xs uppercase tracking-luxury text-muted-foreground mb-6">
          <Link to="/shop" className="hover:text-accent">Shop</Link>
          <span className="mx-2">/</span>
          <Link to={`/category/${product.category}`} className="hover:text-accent capitalize">{product.category}</Link>
        </p>

        <div className="grid md:grid-cols-2 gap-8 md:gap-16">
          {/* Gallery */}
          <div>
            <div className="aspect-[4/5] bg-secondary overflow-hidden rounded-sm hover-zoom">
              <img src={product.images[active]} alt={product.name} className="w-full h-full object-cover" />
            </div>
            <div className="grid grid-cols-4 gap-2 mt-3">
              {product.images.map((img, i) => (
                <button
                  key={i}
                  onClick={() => setActive(i)}
                  className={`aspect-[4/5] overflow-hidden rounded-sm border transition-colors ${
                    active === i ? "border-accent" : "border-transparent hover:border-border"
                  }`}
                >
                  <img src={img} alt="" className="w-full h-full object-cover" />
                </button>
              ))}
            </div>
          </div>

          {/* Info */}
          <div className="md:py-4">
            {product.tags[0] && (
              <p className="text-xs uppercase tracking-luxury text-accent mb-3">{product.tags[0]}</p>
            )}
            <h1 className="font-serif text-4xl md:text-5xl mb-3">{product.name}</h1>
            <p className="text-2xl text-foreground/90 mb-8">{formatINR(product.price)}</p>
            <p className="text-muted-foreground leading-relaxed mb-8">{product.description}</p>

            {/* Size */}
            <div className="mb-8">
              <p className="text-xs uppercase tracking-luxury mb-3">Size</p>
              <div className="flex gap-2">
                {["XS", "S", "M", "L", "XL"].map(s => (
                  <button
                    key={s}
                    className="h-11 w-11 border border-border text-sm hover:border-accent hover:text-accent transition-colors"
                  >
                    {s}
                  </button>
                ))}
              </div>
              <button className="mt-3 text-xs text-muted-foreground story-link">Made to measure available</button>
            </div>

            {/* CTA */}
            <div className="flex gap-3">
              <button className="flex-1 bg-accent text-accent-foreground py-4 text-xs uppercase tracking-luxury hover:bg-foreground transition-colors duration-500">
                Add to bag
              </button>
              <button aria-label="Wishlist" className="border border-border w-14 flex items-center justify-center hover:border-accent hover:text-accent transition-colors">
                <Heart className="h-4 w-4" strokeWidth={1.5} />
              </button>
            </div>

            {/* Meta */}
            <div className="mt-10 pt-8 border-t border-border space-y-4 text-sm">
              <div className="flex justify-between"><span className="text-muted-foreground">Fabric</span><span>{product.fabric}</span></div>
              <div className="flex justify-between"><span className="text-muted-foreground">Crafted in</span><span>Mumbai, India</span></div>
            </div>

            <div className="mt-8 grid grid-cols-3 gap-4 text-xs text-muted-foreground">
              <div className="flex flex-col items-center text-center gap-2"><Truck className="h-4 w-4" /> Free shipping</div>
              <div className="flex flex-col items-center text-center gap-2"><Scissors className="h-4 w-4" /> Free alterations</div>
              <div className="flex flex-col items-center text-center gap-2"><RotateCcw className="h-4 w-4" /> 7-day returns</div>
            </div>
          </div>
        </div>
      </section>

      {related.length > 0 && (
        <section className="container py-24">
          <h2 className="font-serif text-3xl md:text-4xl mb-10 text-center">You may also love</h2>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 md:gap-8">
            {related.map((p, i) => <ProductCard key={p.id} product={p} index={i} />)}
          </div>
        </section>
      )}
    </Layout>
  );
};

export default ProductDetail;
