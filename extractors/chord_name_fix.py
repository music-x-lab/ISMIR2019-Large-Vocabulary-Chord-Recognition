

def fix_mirex_chord_name(chord_name):
    def is_valid_scale(scale_name):
        if(1<=len(scale_name)<=2):
            if('A'<=scale_name[0]<='G'):
                if(len(scale_name)==1 or scale_name[1]=='#' or scale_name[1]=='b'):
                    return True
        return False
    if(chord_name=='N' or chord_name=='X'):
        return chord_name
    if(':' in chord_name):
        return chord_name
    if('/' in chord_name):
        tokens=chord_name.split('/')
        assert(is_valid_scale(tokens[0]))
        return '%s:maj/%s'%(tokens[0],tokens[1])
    assert(is_valid_scale(chord_name))
    return '%s:maj'%(chord_name)

