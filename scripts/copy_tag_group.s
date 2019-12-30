image ref
image imported

string prompt

if(!(getoneimagewithprompt(prompt, "select the reference image", ref))) exit(0)
if(!(getoneimagewithprompt(prompt, "select the imported image", imported))) exit(0)

imagecopycalibrationfrom(imported, ref)
taggroup reftags=ref.imagegettaggroup()
taggroup fittedtags=imported.imagegettaggroup()
taggroupcopytagsfrom(fittedtags, reftags)