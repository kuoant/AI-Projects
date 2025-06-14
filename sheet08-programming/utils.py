# stop execution of the programm until enter is input
def wait_for_input(txt = "Press enter to continue..."):
    print("")
    input(txt)
    print("")

# print state-values in a pretty way
def print_values(inst, values):
    for y in reversed(range(inst.height)):
        print("[ " + "  ".join("{:.2f}".format(values[inst.get_state(x,y)])
            for x in range(inst.width)) + " ]")

# print a policy in a pretty way
def print_policy(inst, policy):
    for y in reversed(range(inst.height)):
        print("[ " + "  ".join("{}".format(policy[inst.get_state(x,y)])
            for x in range(inst.width)) + " ]")
