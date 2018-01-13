from distutils.command.build import build as _build
import subprocess
import setuptools

class build(_build):
    sub_commands = _build.sub_commands + [('CustomCommands', None)]

CUSTOM_COMMANDS = [(["sudo", "apt-get", "update", "-y"],"."),
#                   (["sudo", "apt-get", "install", "-y","libglib2.0-0"],"."),
                   (["sudo", "apt-get", "install", "-y","libgtk2.0-dev"],"."),
                   (["sudo", "apt-get", "install", "-y","pkg-config"],"."),
                   (["pip","install","opencv-python"],".")]


class CustomCommands(setuptools.Command):

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def RunCustomCommand(self, command_list):
        print 'Running command: %s' % command_list[0]
        p = subprocess.Popen(command_list[0], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=command_list[1])
        stdout_data, _ = p.communicate()
        print 'Command output: %s' % stdout_data
        if p.returncode != 0:
            raise RuntimeError('Command %s failed: exit code: %s' % (command_list[0], p.returncode))

    def run(self):
        for command in CUSTOM_COMMANDS:
            self.RunCustomCommand(command)


REQUIRED_PACKAGES = []

setuptools.setup(
    name='preprocesser',
    version='0.1',
    #install_requires=REQUIRED_PACKAGES,
    packages=setuptools.find_packages(),
    cmdclass={
        'build': build,
        'CustomCommands': CustomCommands,
    },
    description='My preprocess application package.'
)
