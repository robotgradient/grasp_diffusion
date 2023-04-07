def generate_obj_urdf(obj_path):
    # creates a urdf object on the fly which is needed as objects can only be spawned from a urdf file
    # returns the path to the urdf file
    obj_path = obj_path[:-4]
    excl_filename = obj_path.split('/')[-1]
    with open(str(obj_path) + '.urdf', 'w') as f:
        f.write('<?xml version="1.0" ?>\n')
        f.write('<robot name="obj.urdf">\n')
        f.write('  <link name="baseLink">\n')
        f.write('    <visual>\n')
        f.write('      <origin rpy="0 0 0" xyz="0 0 0"/>\n')
        f.write('      <geometry>\n')
        f.write('        <mesh filename="' + str(excl_filename) + '.obj" />\n')
        f.write('      </geometry>\n')
        f.write('    </visual>\n')
        f.write('    <collision>\n')
        f.write('      <origin rpy="0 0 0" xyz="0 0 0"/>\n')
        f.write('      <geometry>\n')
        f.write('        <mesh filename="' + str(excl_filename) + '.obj" />\n')
        f.write('      </geometry>\n')
        f.write('    </collision>\n')
        f.write('  </link>\n')
        f.write('</robot>')
    return str(obj_path) + '.urdf'
