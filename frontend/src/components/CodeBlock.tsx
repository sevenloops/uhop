import { useState } from "react";
import { Check, Copy } from "lucide-react";
import { Button } from "./ui/button";

interface CodeBlockProps {
  code: string;
  language?: string;
}

const CodeBlock = ({ code, language = "python" }: CodeBlockProps) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="relative group rounded-lg overflow-hidden border border-border bg-muted/30">
      <div className="flex items-center justify-between px-4 py-2 border-b border-border bg-card/50">
        <span className="text-xs text-muted-foreground font-mono">
          {language}
        </span>
        <Button
          variant="ghost"
          size="sm"
          onClick={handleCopy}
          className="h-7 text-xs"
        >
          {copied ? (
            <>
              <Check className="h-3 w-3 mr-1" />
              Copied
            </>
          ) : (
            <>
              <Copy className="h-3 w-3 mr-1" />
              Copy
            </>
          )}
        </Button>
      </div>
      <pre className="p-4 overflow-x-auto">
        <code className="text-sm font-mono text-foreground">{code}</code>
      </pre>
    </div>
  );
};

export default CodeBlock;
