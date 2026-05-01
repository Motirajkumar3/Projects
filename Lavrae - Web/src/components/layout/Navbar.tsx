import { Link, NavLink, useLocation } from "react-router-dom";
import { useEffect, useState } from "react";
import { Search, ShoppingBag, Menu, X } from "lucide-react";
import { categories } from "@/data/products";

const links = [
  { to: "/", label: "Home" },
  { to: "/new-arrivals", label: "New Arrivals" },
  { to: "/shop", label: "All" },
  ...categories.map(c => ({ to: `/category/${c.slug}`, label: c.name })),
];

const Navbar = () => {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);
  const { pathname } = useLocation();

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => { setOpen(false); }, [pathname]);

  return (
    <header
      className={`fixed top-0 inset-x-0 z-50 transition-all duration-500 ${
        scrolled ? "bg-background/85 backdrop-blur-md border-b border-border/60" : "bg-transparent"
      }`}
    >
      <div className="container flex items-center justify-between h-16 md:h-20">
        <button
          aria-label="Menu"
          className="md:hidden -ml-2 p-2"
          onClick={() => setOpen(v => !v)}
        >
          {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>

        <Link to="/" className="font-serif text-2xl md:text-3xl tracking-tight">
          V<span className="text-accent">·</span>tailers
        </Link>

        <nav className="hidden md:flex items-center gap-8 text-xs uppercase tracking-luxury">
          {links.map(l => (
            <NavLink
              key={l.to}
              to={l.to}
              end={l.to === "/"}
              className={({ isActive }) =>
                `story-link transition-colors ${isActive ? "text-accent" : "text-foreground/80 hover:text-foreground"}`
              }
            >
              {l.label}
            </NavLink>
          ))}
        </nav>

        <div className="flex items-center gap-1 md:gap-3">
          <button aria-label="Search" className="p-2 hover:text-accent transition-colors">
            <Search className="h-4.5 w-4.5" strokeWidth={1.4} />
          </button>
          <button aria-label="Bag" className="p-2 hover:text-accent transition-colors relative">
            <ShoppingBag className="h-4.5 w-4.5" strokeWidth={1.4} />
            <span className="absolute -top-0.5 -right-0.5 h-4 w-4 rounded-full bg-accent text-accent-foreground text-[10px] flex items-center justify-center font-medium">0</span>
          </button>
        </div>
      </div>

      {/* Mobile drawer */}
      <div
        className={`md:hidden overflow-hidden transition-all duration-500 bg-background border-b border-border ${
          open ? "max-h-[80vh]" : "max-h-0"
        }`}
      >
        <nav className="container flex flex-col py-6 gap-4 text-sm uppercase tracking-luxury">
          {links.map(l => (
            <NavLink
              key={l.to}
              to={l.to}
              end={l.to === "/"}
              className={({ isActive }) => (isActive ? "text-accent" : "text-foreground/80")}
            >
              {l.label}
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  );
};

export default Navbar;
