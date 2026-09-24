-- Pandoc filter for the submission manuscript (docx only). It changes presentation, not text:
--  1. Pandoc's default docx template prints only the title and author names, so the affiliation,
--     corresponding author and keywords from the YAML front matter are added as visible paragraphs
--     above the abstract.
--  2. Each figure is written as `![Figure N](file)` followed by an italic paragraph holding the real
--     caption, so pandoc would print the bare "Figure N" alt text as a caption and then the real one.
--     The bare caption is dropped and the italic paragraph is set in Word's "Image Caption" style.
--  3. The bold "Table N." paragraph that precedes each table is set in "Table Caption" style.

local stringify = pandoc.utils.stringify

local function styled(style, blocks)
  return pandoc.Div(blocks, pandoc.Attr("", {}, { ["custom-style"] = style }))
end

function Blocks(blocks)
  if FORMAT ~= "docx" then return nil end
  local out, i = pandoc.List(), 1
  while i <= #blocks do
    local b, nxt = blocks[i], blocks[i + 1]
    if b.t == "Figure" and stringify(b.caption.long):match("^Figure %d+$") then
      b.caption = pandoc.Caption({}, {})
      out:insert(b)
      if nxt and nxt.t == "Para" and #nxt.content == 1 and nxt.content[1].t == "Emph" then
        out:insert(styled("Image Caption", { pandoc.Para(nxt.content[1].content) }))
        i = i + 1
      end
    elseif b.t == "Para" and nxt and nxt.t == "Table" and stringify(b):match("^Table %d+%.") then
      out:insert(styled("Table Caption", { b }))
    else
      out:insert(b)
    end
    i = i + 1
  end
  return out
end

function Pandoc(doc)
  if FORMAT ~= "docx" then return nil end
  local m = doc.meta
  local front = {}
  if m.affiliation then
    table.insert(front, styled("Author", { pandoc.Para({ pandoc.Str(stringify(m.affiliation)) }) }))
  end
  if m["corresponding-author"] then
    table.insert(front, styled("Author", { pandoc.Para({
      pandoc.Str("Corresponding author:"), pandoc.Space(), pandoc.Str(stringify(m["corresponding-author"])),
    }) }))
  end
  if m.keywords then
    local kws = {}
    for _, k in ipairs(m.keywords) do table.insert(kws, stringify(k)) end
    table.insert(front, pandoc.Para({ pandoc.Strong({ pandoc.Str("Keywords:") }), pandoc.Space(),
                                      pandoc.Str(table.concat(kws, "; ")) }))
  end
  for i = #front, 1, -1 do table.insert(doc.blocks, 1, front[i]) end
  return doc
end
