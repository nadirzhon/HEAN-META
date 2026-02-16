import clsx from "clsx";

interface HeadingProps {
  as?: "h1" | "h2" | "h3";
  children: React.ReactNode;
  className?: string;
  gradient?: boolean;
}

export default function Heading({
  as: Tag = "h2",
  children,
  className,
  gradient = false,
}: HeadingProps) {
  const sizes = {
    h1: "text-5xl md:text-7xl font-bold tracking-tight leading-[1.1]",
    h2: "text-3xl md:text-5xl font-bold tracking-tight leading-tight",
    h3: "text-xl md:text-2xl font-semibold tracking-tight",
  };

  return (
    <Tag
      className={clsx(
        "font-[family-name:var(--font-heading)]",
        sizes[Tag],
        gradient ? "gradient-text" : "text-supernova",
        className
      )}
    >
      {children}
    </Tag>
  );
}
